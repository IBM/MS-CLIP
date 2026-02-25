# Copyright 2025 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import pandas as pd
from pathlib import Path
from collections.abc import Callable

from msclip.inference.utils import (
    build_model,
    preprocess_and_stack,
    load_image_paths,
    load_queries,
)

if torch.cuda.is_available():
    default_device = "cuda"
elif torch.mps.is_available():
    default_device = "mps"
else:
    default_device = "cpu"


def run_inference_retrieval(
        model: torch.nn.Module = None,
        preprocess: Callable = None,
        tokenizer: Callable = None,
        model_name: str = "Llama3-MS-CLIP-Base",
        pretrained: bool = True,
        ckpt_path: str = None,
        image_path: str | list[str] = None,
        queries: list[str] = None,
        queries_file: str = None,
        top_k: int = 5,
        save_path: str = None,
        device: str = None,
        verbose: bool = True,
):
    device = device or default_device
    if model is None or preprocess is None or tokenizer is None:
        # Load model from HF
        model, preprocess, tokenizer = build_model(model_name, pretrained, ckpt_path)

    model.to(device)

    if isinstance(image_path, list):
        image_paths = image_path
    else:
        image_paths = load_image_paths(image_path)

    if not queries and not queries_file:
        raise ValueError("Please provide queries as list of strings or a queries_file")
    queries = queries or load_queries(queries_file)

    # Encode image features
    if isinstance(image_paths, str) and os.path.isdir(image_paths):
        image_paths = [
            os.path.join(image_paths, f) for f in os.listdir(image_paths)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff", ".npy", ".npz"))
        ]
    image_tensor = preprocess_and_stack(image_paths, preprocess, is_ms=model.channels > 3)
    image_tensor = image_tensor.to(device)

    with torch.no_grad(), torch.autocast(device_type=device):
        image_features = model.inference_vision(image_tensor)  # [B, D]
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        # Encode text
        text_tokens = tokenizer(queries).to(device)
        text_features = model.inference_text(text_tokens)  # [T, D]
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        # Similarity: [num_images, num_queries]
        similarity = image_features @ text_features.T  # [B, T]

    results_df = pd.DataFrame(columns=["Query", "Rank", "Image", "Similarity"])
    for i, query in enumerate(queries):
        topk = torch.topk(similarity[:, i], k=top_k)
        topk_indices = topk.indices.tolist()
        topk_scores = topk.values.tolist()

        topk_items = [
            (query, rank + 1, os.path.basename(image_paths[idx]), score)
            for rank, (idx, score) in enumerate(zip(topk_indices, topk_scores))
        ]

        topk_items = pd.DataFrame(topk_items, columns=["Query", "Rank", "Image", "Similarity"])
        results_df = pd.concat([results_df, topk_items])

    results_df = results_df.set_index(["Query", "Rank"])

    if verbose:
        print(f"Text-to-Image Retrieval (Top-{top_k}) Results:")
        print(results_df)

    if save_path:
        save_path = Path(save_path).with_suffix(".csv")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(save_path)
        if verbose:
            print(f"Saved text-to-image retrieval results to {save_path}")

    return results_df
