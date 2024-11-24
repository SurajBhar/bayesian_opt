import os
import pathlib
import torch
from typing import Tuple, Dict, List
from torch.utils.data import Dataset
from PIL import Image
from typing import Tuple

import os
import sys
import getpass

def setup_ccname():
    user=getpass.getuser()
    # check if k5start is running, exit otherwise
    try:
        pid=open("/tmp/k5pid_"+user).read().strip()
        os.kill(int(pid), 0)
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nk5start not running!\n")
        sys.exit(1)
    try:
        ccname=open("/tmp/kccache_"+user).read().split("=")[1].strip()
        os.environ['KRB5CCNAME']=ccname
    except:
        sys.stderr.write("Unable to setup KRB5CCNAME!\nmaybe k5start not running?\n")
        sys.exit(1)


def setup_library_paths():
    """Configures additional library paths required for GPU computation.
    Note: These paths are specific to my local server.
          You might not need to define them.

    Adds specified library paths to the LD_LIBRARY_PATH environment variable
    if they are not already included, ensuring GPU dependencies are located.
    """
    library_paths = [
        "/usr/lib/xorg-nvidia-525.116.04/lib/x86_64-linux-gnu",
        "/usr/lib/xorg/lib/x86_64-linux-gnu",
        "/usr/lib/xorg-nvidia-535.113.01/lib/x86_64-linux-gnu",
    ]
    current_ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = [path for path in library_paths if path not in current_ld_library_path]
    os.environ["LD_LIBRARY_PATH"] = ":".join(new_paths + [current_ld_library_path])
    print("Updated LD_LIBRARY_PATH:")
    print(os.environ["LD_LIBRARY_PATH"])

# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None, target_transform=None) -> None:
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*/*.png")) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        self.target_transform = target_transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = self.find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        image = Image.open(image_path).convert("RGB")
        return image
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Make function to find classes in target directory
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folder names in a target directory.
        
        Assumes target directory is in standard image classification format.

        Args:
            directory (str): target directory to load classnames from.

        Returns:
            Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
        
        Example:
            find_classes("food_images/train")
            >>> (["class_1", "class_2"], {"class_1": 0, ...})
        """
        # 1. Get the class names by scanning the target directory
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        
        # 2. Raise an error if class names not found
        if not classes:
            raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
            
        # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    # 7. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        image = self.load_image(index)
        # Get the class name from grandparent directory
        class_name  = self.paths[index].parent.parent.name # expects path in data_folder/class_name/chunk_0/image.png
        class_idx = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            class_idx = self.target_transform(class_idx)
        return image, class_idx
    
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Sequence
import torch
from torchvision import transforms


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


class MaybeToTensor(transforms.ToTensor):
    """
    Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor, or keep as is if already a tensor.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image, numpy.ndarray or torch.tensor): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return super().__call__(pic)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Normalize:
    return transforms.Normalize(mean=mean, std=std)


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = 224,
    interpolation=transforms.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [transforms.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(transforms.RandomHorizontalFlip(hflip_prob))
    transforms_list.extend(
        [
            MaybeToTensor(),
            make_normalize_transform(mean=mean, std=std),
        ]
    )
    return transforms.Compose(transforms_list)


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = 256,
    interpolation=transforms.InterpolationMode.BICUBIC,
    crop_size: int = 224,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> transforms.Compose:
    transforms_list = [
        transforms.Resize(resize_size, interpolation=interpolation),
        transforms.CenterCrop(crop_size),
        MaybeToTensor(),
        make_normalize_transform(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)


######## Linear Classifier #######################
###################################################
import torch.nn as nn

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, backbone, in_features=768, num_classes=34):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.linear = nn.Linear(in_features, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        features = self.backbone(x)
        output = self.linear(features)
        return output
    