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

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


class RESISC45(ImageFolder):
    """ Image Scene Classification dataset from 'Remote Sensing Image
    Scene Classification: Benchmark and State of the Art', Cheng at al. (2017)
    https://arxiv.org/abs/1703.00121

    'We propose a large-scale dataset, termed "NWPU-RESISC45", which is a publicly
    available benchmark for REmote Sensing Image Scene Classification (RESISC), created
    by Northwestern Polytechnical University (NWPU). This dataset contains 31,500 images,
    covering 45 scene classes with 700 images in each class. The proposed NWPU-RESISC45 (i)
    is large-scale on the scene classes and the total image number, (ii) holds big variations
    in translation, spatial resolution, viewpoint, object pose, illumination, background, and
    occlusion, and (iii) has high within-class diversity and between-class similarity.'
    """

    def __init__(
            self,
            root: str = "/",
            transform: T.Compose = T.Compose([T.ToTensor()])
    ):
        super().__init__(
            root=root,
            transform=transform
        )
