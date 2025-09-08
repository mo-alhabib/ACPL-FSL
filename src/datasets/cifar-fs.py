from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle as pkl


def _npz_get(arr_dict, keys):
    """Robustly fetch an array from an npz mapping."""
    for k in keys:
        if k in arr_dict:
            return arr_dict[k]
    raise KeyError(f"None of the keys {keys} found in npz file.")


class NonEpisodicCIFARFS(Dataset):
    """Non-episodic CIFAR-FS loader (NPZ)."""
    tasks_type = "clss"
    name = "cifarfs"
    split_paths = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    episodic = False
    c = 3
    h = 32
    w = 32

    def __init__(self, data_root, split, transforms, **kwargs):
        """
        Args:
            data_root: directory containing 'cifar-fs-<split>.npz'
            split: one of {'train','val'/'valid','test'}
            transforms: callable applied to HWC numpy image
        """
        self.data_root = os.path.join(data_root, "cifar-fs-%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        # Accept common key variants
        self.features = _npz_get(data, ["features", "images", "data"])
        self.labels = _npz_get(data, ["targets", "labels"])
        self.transforms = transforms

    def next_run(self):
        pass

    def __getitem__(self, item):
        image = self.transforms(self.features[item])
        image = image * 2 - 1
        return image, self.labels[item]

    def __len__(self):
        return len(self.features)


class RotatedNonEpisodicCIFARFS(Dataset):
    """Rotation-augmented non-episodic CIFAR-FS (NPZ)."""
    tasks_type = "clss"
    name = "cifarfs"
    split_paths = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    episodic = False
    c = 3
    h = 32
    w = 32

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """
        Args:
            rotation_labels: indices for self-supervised rotation heads (0°/90°/180°/270°)
        """
        self.data_root = os.path.join(data_root, "cifar-fs-%s.npz")
        data = np.load(self.data_root % self.split_paths[split])
        self.features = _npz_get(data, ["features", "images", "data"])
        self.labels = _npz_get(data, ["targets", "labels"])
        self.transforms = transforms
        self.size = len(self.features)
        self.rotation_labels = rotation_labels

    def next_run(self):
        pass

    def rotate_img(self, img, rot):
        if rot == 0:
            return img
        elif rot == 90:
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:
            return np.fliplr(np.flipud(img))
        elif rot == 270:
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = self.features[item]
        if np.random.randint(2):
            image = np.fliplr(image).copy()
        cat = [self.transforms(image)]
        if len(self.rotation_labels) > 1:
            image_90 = self.transforms(self.rotate_img(image, 90))
            image_180 = self.transforms(self.rotate_img(image, 180))
            image_270 = self.transforms(self.rotate_img(image, 270))
            cat.extend([image_90, image_180, image_270])
        images = torch.stack(cat) * 2 - 1
        return images, torch.ones(len(self.rotation_labels), dtype=torch.long) * int(self.labels[item]), torch.LongTensor(self.rotation_labels)

    def __len__(self):
        return self.size


class RotatedNonEpisodicCIFARFSPkl(Dataset):
    """Rotation-augmented non-episodic CIFAR-FS (PKL)."""
    tasks_type = "clss"
    name = "cifarfs"
    split_paths = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    episodic = False
    c = 3
    h = 32
    w = 32

    def __init__(self, data_root, split, transforms, rotation_labels=[0, 1, 2, 3], **kwargs):
        """
        Expects one of:
            - dict with 'image_data' + 'class_dict' (miniImageNet-style)
            - dict with 'data' + 'labels' (CIFAR-style)
        """
        self.data_root = os.path.join(data_root, "cifar-fs-cache-%s.pkl")
        with open(self.data_root % self.split_paths[split], 'rb') as infile:
            data = pkl.load(infile)

        if "image_data" in data and "class_dict" in data:
            self.features = data["image_data"]
            label_names = data["class_dict"].keys()
            self.labels = np.zeros((self.features.shape[0],), dtype=int)
            for i, name in enumerate(sorted(label_names)):
                self.labels[np.array(data['class_dict'][name])] = i
        elif "data" in data and "labels" in data:
            self.features = np.array(data["data"])
            self.labels = np.array(data["labels"])
        else:
            raise KeyError("Unsupported PKL schema. Expect ('image_data','class_dict') or ('data','labels').")

        del data
        self.transforms = transforms
        self.size = len(self.features)
        self.rotation_labels = rotation_labels

    def next_run(self):
        pass

    def rotate_img(self, img, rot):
        if rot == 0:
            return img
        elif rot == 90:
            return np.flipud(np.transpose(img, (1, 0, 2)))
        elif rot == 180:
            return np.fliplr(np.flipud(img))
        elif rot == 270:
            return np.transpose(np.flipud(img), (1, 0, 2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

    def __getitem__(self, item):
        image = self.features[item]
        if np.random.randint(2):
            image = np.fliplr(image).copy()
        cat = [self.transforms(image)]
        if len(self.rotation_labels) > 1:
            image_90 = self.transforms(self.rotate_img(image, 90))
            image_180 = self.transforms(self.rotate_img(image, 180))
            image_270 = self.transforms(self.rotate_img(image, 270))
            cat.extend([image_90, image_180, image_270])
        images = torch.stack(cat) * 2 - 1
        return images, torch.ones(len(self.rotation_labels), dtype=torch.long) * int(self.labels[item]), torch.LongTensor(self.rotation_labels)

    def __len__(self):
        return self.size
