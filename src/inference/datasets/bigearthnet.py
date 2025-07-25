# Code adapted from https://github.com/microsoft/torchgeo
#
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet dataset."""

import glob
import json
import os
from typing import Callable, Optional

import numpy as np
import rasterio
import torch

from rasterio.enums import Resampling
from torch import Tensor
from torchvision import transforms
from src.inference.datasets.geo import NonGeoDataset
from src.inference.datasets.utils import download_url, extract_archive, sort_sentinel2_bands
from src.inference.utils import Unsqueeze, SelectChannels, AddMeanChannels
from src.inference.utils import SelectChannels, Unsqueeze, DictTransforms, ConvertType, AddMeanChannels


class BigEarthNet(NonGeoDataset):
    """BigEarthNet dataset.

    The `BigEarthNet <https://bigearth.net/>`__
    dataset is a dataset for multilabel remote sensing image scene classification.

    Dataset features:

    * 590,326 patches from 125 Sentinel-1 and Sentinel-2 tiles
    * Imagery from tiles in Europe between Jun 2017 - May 2018
    * 12 spectral bands with 10-60 m per pixel resolution (base 120x120 px)
    * 2 synthetic aperture radar bands (120x120 px)
    * 43 or 19 scene classes from the 2018 CORINE Land Cover database (CLC 2018)

    Dataset format:

    * images are composed of multiple single channel geotiffs
    * labels are multiclass, stored in a single json file per image
    * mapping of Sentinel-1 to Sentinel-2 patches are within Sentinel-1 json files
    * Sentinel-1 bands: (VV, VH)
    * Sentinel-2 bands: (B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * All bands: (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    * Sentinel-2 bands are of different spatial resolutions and upsampled to 10m

    Dataset classes (43):

    0. Continuous urban fabric
    1. Discontinuous urban fabric
    2. Industrial or commercial units
    3. Road and rail networks and associated land
    4. Port areas
    5. Airports
    6. Mineral extraction sites
    7. Dump sites
    8. Construction sites
    9. Green urban areas
    10. Sport and leisure facilities
    11. Non-irrigated arable land
    12. Permanently irrigated land
    13. Rice fields
    14. Vineyards
    15. Fruit trees and berry plantations
    16. Olive groves
    17. Pastures
    18. Annual crops associated with permanent crops
    19. Complex cultivation patterns
    20. Land principally occupied by agriculture, with significant
        areas of natural vegetation
    21. Agro-forestry areas
    22. Broad-leaved forest
    23. Coniferous forest
    24. Mixed forest
    25. Natural grassland
    26. Moors and heathland
    27. Sclerophyllous vegetation
    28. Transitional woodland/shrub
    29. Beaches, dunes, sands
    30. Bare rock
    31. Sparsely vegetated areas
    32. Burnt areas
    33. Inland marshes
    34. Peatbogs
    35. Salt marshes
    36. Salines
    37. Intertidal flats
    38. Water courses
    39. Water bodies
    40. Coastal lagoons
    41. Estuaries
    42. Sea and ocean

    Dataset classes (19):

    0. Urban fabric
    1. Industrial or commercial units
    2. Arable land
    3. Permanent crops
    4. Pastures
    5. Complex cultivation patterns
    6. Land principally occupied by agriculture, with significant
       areas of natural vegetation
    7. Agro-forestry areas
    8. Broad-leaved forest
    9. Coniferous forest
    10. Mixed forest
    11. Natural grassland and sparsely vegetated areas
    12. Moors, heathland and sclerophyllous vegetation
    13. Transitional woodland, shrub
    14. Beaches, dunes, sands
    15. Inland wetlands
    16. Coastal wetlands
    17. Inland waters
    18. Marine waters

    The source for the above dataset classes, their respective ordering, and
    43-to-19-class mappings can be found here:

    * https://git.tu-berlin.de/rsim/BigEarthNet-S2_19-classes_models/-/blob/master/label_indices.json

    If you use this dataset in your research, please cite the following paper:

    * https://doi.org/10.1109/IGARSS.2019.8900532

    """  # noqa: E501

    class_sets = {
        19: [
            "Urban fabric",
            "Industrial or commercial units",
            "Arable land",
            "Permanent crops",
            "Pastures",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland and sparsely vegetated areas",
            "Moors, heathland and sclerophyllous vegetation",
            "Transitional woodland, shrub",
            "Beaches, dunes, sands",
            "Inland wetlands",
            "Coastal wetlands",
            "Inland waters",
            "Marine waters",
            
        ],
        43: [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture, with significant areas of"
            " natural vegetation",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean",
        ],
    }

    

    label_converter = {
        0: 0,
        1: 0,
        2: 1,
        11: 2,
        12: 2,
        13: 2,
        14: 3,
        15: 3,
        16: 3,
        18: 3,
        17: 4,
        19: 5,
        20: 6,
        21: 7,
        22: 8,
        23: 9,
        24: 10,
        25: 11,
        31: 11,
        26: 12,
        27: 12,
        28: 13,
        29: 14,
        33: 15,
        34: 15,
        35: 16,
        36: 16,
        38: 17,
        39: 17,
        40: 18,
        41: 18,
        42: 18,
    }

    splits_metadata = {
        "train": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/train.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-train.csv",
            "md5": "623e501b38ab7b12fe44f0083c00986d",
        },
        "val": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/val.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-val.csv",
            "md5": "22efe8ed9cbd71fa10742ff7df2b7978",
        },
        "test": {
            "url": "https://git.tu-berlin.de/rsim/BigEarthNet-MM_19-classes_models/-/raw/9a5be07346ab0884b2d9517475c27ef9db9b5104/splits/test.csv?inline=false",  # noqa: E501
            "filename": "bigearthnet-test.csv",
            "md5": "697fb90677e30571b9ac7699b7e5b432",
        },
    }
    metadata = {
        "s1": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S1-v1.0.tar.gz",
            "md5": "94ced73440dea8c7b9645ee738c5a172",
            "filename": "BigEarthNet-S1-v1.0.tar.gz",
            "directory": "sentinel-1"
            #"directory": "BigEarthNet-S1-v1.0",
        },
        "s2": {
            "url": "https://bigearth.net/downloads/BigEarthNet-S2-v1.0.tar.gz",
            "md5": "5a64e9ce38deb036a435a7b59494924c",
            "filename": "BigEarthNet-S2-v1.0.tar.gz",
            "directory": "sentinel-2"
            #"directory": "BigEarthNet-v1.0",
        },
    }
    image_size = (120, 120)

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        bands: str = "all",
        num_classes: int = 19,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
        other_features = False,
    ) -> None:
        """Initialize a new BigEarthNet dataset instance.

        Args:
            root: root directory where dataset can be found
            split: train/val/test split to load
            bands: load Sentinel-1 bands, Sentinel-2, or both. one of {s1, s2, all}
            num_classes: number of classes to load in target. one of {19, 43}
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            download: if True, download dataset and store it in the root directory
            checksum: if True, check the MD5 of the downloaded files (may be slow)
        """
        assert split in self.splits_metadata
        assert bands in ["s1", "s2", "all"]
        assert num_classes in [43, 19]
        self.root = root
        self.split = split
        self.bands = bands
        self.num_classes = num_classes   #remove if no ther class
        self.transforms = transforms
        self.download = download
        self.checksum = checksum
        self.class2idx = {c: i for i, c in enumerate(self.class_sets[43])}
        self._verify()
        self.folders = self._load_folders()
        self.other_features = other_features

        if self.other_features and "Other features" not in self.class_sets[19]:
            self.class_sets[19].append("Other features")
            self.class_sets[43].append("Other features")
            self.num_classes = self.num_classes + 1

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image = self._load_image(index)
        label = self._load_target(index)
        sample: dict[str, Tensor] = {"image": image, "label": label}
        if self.transforms is not None:
            sample = self.transforms(sample)      
        if len(sample["image"].shape)==4:
            sample["image"] = sample["image"].squeeze(1)


        return sample["image"], sample["label"]

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.folders)

    def _load_folders(self) -> list[dict[str, str]]:
        """Load folder paths.

        Returns:
            list of dicts of s1 and s2 folder paths
        """
        filename = self.splits_metadata[self.split]["filename"]
        dir_s1 = self.metadata["s1"]["directory"]
        dir_s2 = self.metadata["s2"]["directory"]

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            pairs = [line.split(",") for line in lines]

        
        

        folders = [
            {
                "s1": os.path.join(self.root, dir_s1, pair[0]),
                "s2": os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders

    def _load_paths(self, index: int) -> list[str]:
        """Load paths to band files.

        Args:
            index: index to return

        Returns:
            list of file paths
        """
        if self.bands == "all":
            folder_s1 = self.folders[index]["s1"]
            folder_s2 = self.folders[index]["s2"]
            paths_s1 = glob.glob(os.path.join(folder_s1, "*.tif"))
            paths_s2 = glob.glob(os.path.join(folder_s2, "*.tif"))
            paths_s1 = sorted(paths_s1)
            paths_s2 = sorted(paths_s2, key=sort_sentinel2_bands)
            paths = paths_s1 + paths_s2
        elif self.bands == "s1":
            folder = self.folders[index]["s1"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths)
        else:
            folder = self.folders[index]["s2"]
            paths = glob.glob(os.path.join(folder, "*.tif"))
            paths = sorted(paths, key=sort_sentinel2_bands)

        return paths

    def _load_image(self, index: int) -> Tensor:
        """Load a single image.

        Args:
            index: index to return

        Returns:
            the raster image or target
        """
        paths = self._load_paths(index)
        images = []
        for path in paths:
            # Bands are of different spatial resolutions
            # Resample to (120, 120)
            with rasterio.open(path) as dataset:
                array = dataset.read(
                    indexes=1,
                    out_shape=self.image_size,
                    out_dtype="int32",
                    resampling=Resampling.bilinear,
                )
                images.append(array)
        arrays: "np.typing.NDArray[np.int_]" = np.stack(images, axis=0)
        tensor = torch.from_numpy(arrays).float()
        return tensor

    def _load_target(self, index: int) -> Tensor:
        """Load the target mask for a single image.

        Args:
            index: index to return

        Returns:
            the target label
        """
        if self.bands == "s2":
            folder = self.folders[index]["s2"]
        else:
            folder = self.folders[index]["s1"]

        path = glob.glob(os.path.join(folder, "*.json"))[0]
        with open(path) as f:
            labels = json.load(f)["labels"]

        # labels -> indices
        indices = [self.class2idx[label] for label in labels]

        # Map 43 to 19/20 class labels
        if self.num_classes == 19 or self.num_classes == 20: #remove 20 if you have no other class
            indices_optional = [self.label_converter.get(idx) for idx in indices]
            indices = [idx for idx in indices_optional if idx is not None]

        target = torch.zeros(self.num_classes, dtype=torch.long)
        target[indices] = 1
        return target

    def _verify(self) -> None:
        """Verify the integrity of the dataset.

        Raises:
            RuntimeError: if ``download=False`` but dataset is missing or checksum fails
        """
        keys = ["s1", "s2"] if self.bands == "all" else [self.bands]
        urls = [self.metadata[k]["url"] for k in keys]
        md5s = [self.metadata[k]["md5"] for k in keys]
        filenames = [self.metadata[k]["filename"] for k in keys]
        directories = [self.metadata[k]["directory"] for k in keys]
        urls.extend([self.splits_metadata[k]["url"] for k in self.splits_metadata])
        md5s.extend([self.splits_metadata[k]["md5"] for k in self.splits_metadata])
        filenames_splits = [
            self.splits_metadata[k]["filename"] for k in self.splits_metadata
        ]
        filenames.extend(filenames_splits)

        # Check if the split file already exist
        exists = []
        for filename in filenames_splits:
            exists.append(os.path.exists(os.path.join(self.root, filename)))

        # Check if the files already exist
        for directory in directories:
            exists.append(os.path.exists(os.path.join(self.root, directory)))

        if all(exists):
            return

        # Check if zip file already exists (if so then extract)
        exists = []
        for filename in filenames:
            filepath = os.path.join(self.root, filename)
            if os.path.exists(filepath):
                exists.append(True)
                self._extract(filepath)
            else:
                exists.append(False)

        if all(exists):
            return

        # Check if the user requested to download the dataset
        if not self.download:
            raise RuntimeError(
                "Dataset not found in `root` directory and `download=False`, "
                "either specify a different `root` directory or use `download=True` "
                "to automatically download the dataset."
            )

        # Download and extract the dataset
        for url, filename, md5 in zip(urls, filenames, md5s):
            self._download(url, filename, md5)
            filepath = os.path.join(self.root, filename)
            self._extract(filepath)

    def _download(self, url: str, filename: str, md5: str) -> None:
        """Download the dataset.

        Args:
            url: url to download file
            filename: output filename to write downloaded file
            md5: md5 of downloaded file
        """
        if not os.path.exists(filename):
            download_url(
                url, self.root, filename=filename, md5=md5 if self.checksum else None
            )

    def _extract(self, filepath: str) -> None:
        """Extract the dataset.

        Args:
            filepath: path to file to be extracted
        """
        if not filepath.endswith(".csv"):
            extract_archive(filepath)

    def _onehot_labels_to_names(
        self, label_mask: "np.typing.NDArray[np.bool_]"
    ) -> list[str]:
        """Gets a list of class names given a label mask.

        Args:
            label_mask: a boolean mask corresponding to a set of labels or predictions

        Returns
            a list of class names corresponding to the input mask
        """
        labels = []
        for i, mask in enumerate(label_mask):
            if mask:
                labels.append(self.class_sets[self.num_classes][i])
        return labels


    
def init_bigearthnet(path, bands, normalize, num_classes, means, stds, other_features, rgb = False, *args, **kwargs):
    """
    Init BigEarthNet dataset, with S2 data and 43 classes as default.
    """
    # Get dataset parameters
    split = 'test'
    satellite = 's2'

    # Get BigEarthNet directory
    bigearthnet_dir = path
    # Check if data is downloaded
    # assert os.path.isdir(os.path.join(bigearthnet_dir, BigEarthNet.metadata['s2']['directory'])), \
    #     "Download BigEarthNet with `sh datasets/bigearthnet_download.sh` or specify the DATA_DIR via a env variable."

    # Init transforms
    if rgb:
        image_transforms = [
        SelectChannels(bands),
        ConvertType(torch.float),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),
        transforms.Lambda(lambda x: x.numpy()),
        transforms.Lambda(lambda x: x.astype(np.float32)  / 2000),
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),
        
        transforms.ToTensor(),  #for rgb the values are scaled but not for ms
        transforms.Resize(
                size=224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        transforms.CenterCrop(224),
        transforms.Normalize(mean= means, std = stds) 
        ]
    else:
        image_transforms = [
        SelectChannels(bands),
        ConvertType(torch.float),
        transforms.Resize(size=224, antialias=True),
    ]

        if normalize:
            # Normalize images
            image_transforms.append(transforms.Normalize(mean=means, std=stds))
            image_transforms.append(Unsqueeze(dim=1))  # add time dim

    ben_transforms = DictTransforms({'image': transforms.Compose(image_transforms)})

    # Init dataset
    dataset = BigEarthNet(
        root=bigearthnet_dir,
        split=split,
        bands=satellite,
        num_classes=num_classes,
        transforms=ben_transforms,
        other_features=other_features
    )

    return dataset