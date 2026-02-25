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
from tabulate import tabulate
from collections.abc import Callable

from msclip.inference.utils import (
    build_model,
    preprocess_and_stack,
    load_classes,
    load_templates,
    load_image_paths,
)
from msclip.inference.clip_benchmark.metrics.zeroshot_eval import zero_shot_classifier

if torch.cuda.is_available():
    default_device = "cuda"
elif torch.mps.is_available():
    default_device = "mps"
else:
    default_device = "cpu"


def run_inference_classification(
        model: torch.nn.Module = None,
        preprocess: Callable = None,
        tokenizer: Callable = None,
        model_name: str = "Llama3-MS-CLIP-Base",
        pretrained: bool = True,
        ckpt_path: str = None,
        image_path: str | list[str] = None,
        class_names: list[str] = None,
        classes_file: str = None,
        save_path: str = None,
        device: str = None,
        templates: str = "msclip",
        verbose: bool = True,
):
    device = device or default_device
    if model is None or preprocess is None or tokenizer is None:
        # Load model from HF
        model, preprocess, tokenizer = build_model(model_name, pretrained, ckpt_path, device)

    model.to(device)

    if isinstance(image_path, list):
        image_paths = image_path
    else:
        image_paths = load_image_paths(image_path)

    # Class and prompt templates
    templates = load_templates(templates)

    if not class_names and not classes_file:
        raise ValueError("Please provide class_names as list of strings or a classes_file")
    classnames = class_names or load_classes(classes_file)

    classifier = zero_shot_classifier(model, tokenizer, classnames, templates, device=device)

    results = []

    image_tensor = preprocess_and_stack(image_paths, preprocess, is_ms=model.channels > 3)  # [B, C, H, W]
    image_tensor = image_tensor.to(device)

    with torch.no_grad(), torch.autocast(device_type=device):
        image_features = model.inference_vision(image_tensor)  # [B, D]
        image_features = torch.nn.functional.normalize(image_features, dim=-1)

        logits = 100. * image_features @ classifier  # [B, num_classes]
        probs = logits.softmax(dim=-1).to(torch.float32).cpu().numpy()  # [B, num_classes]

    for path, prob in zip(image_paths, probs):
        results.append((os.path.basename(path), prob))

    # Format nicely into a DataFrame
    if class_names is None:
        class_names = [f"Class {i}" for i in range(probs.shape[1])]

    results_df = pd.DataFrame([dict(zip(class_names, prob)) for _, prob in results])
    results_df.insert(0, "Class", results_df.idxmax(axis=1))
    results_df.insert(0, "Image", [name for name, _ in results])

    if verbose:
        print(f"Zero-Shot Classification Results:")
        print(tabulate(results_df, headers="keys", tablefmt="fancy_grid", floatfmt=".3f"))

    if save_path:
        save_path = Path(save_path).with_suffix(".csv")
        save_path.parent.mkdir(exist_ok=True, parents=True)
        results_df.to_csv(save_path)
        if verbose:
            print(f"Saved zero-shot classification results to {save_path}")

    return results_df
