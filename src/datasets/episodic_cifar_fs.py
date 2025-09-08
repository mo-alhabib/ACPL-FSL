import numpy as np
import torch
import torchvision
import os
from src.datasets.episodic_dataset import EpisodicDataset, FewShotSampler
import pickle as pkl

def _get(d, keys):
    """Return the first present key from `keys` in dict-like `d`."""
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of the keys {keys} found.")

# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicCIFARFS(EpisodicDataset):
    tasks_type = "clss"
    name = "cifarfs"
    episodic = True
    split_paths = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    # CIFAR-FS native resolution is 32x32; resize in `transforms` if desired.
    # c = 3
    # h = 32
    # w = 32

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Args:
            split: data split
            sampler: FewShotSampler instance
            size: number of tasks to generate (int)
        """
        self.data_root = os.path.join(data_root, "cifar-fs-%s.npz")
        self.split = split
        data = np.load(self.data_root % self.split_paths[split])
        self.features = _get(data, ["features", "images", "data"])
        labels = _get(data, ["targets", "labels"])
        del data
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()


# Inherit order is important, FewShotDataset constructor is prioritary
class EpisodicCIFARFSPkl(EpisodicDataset):
    tasks_type = "clss"
    name = "cifarfs"
    episodic = True
    split_paths = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    # c = 3
    # h = 32
    # w = 32

    def __init__(self, data_root, split, sampler, size, transforms):
        """ Constructor

        Supports either:
          - miniImageNet-style cache: {'image_data', 'class_dict'}
          - CIFAR-style cache: {'data', 'labels'}
        """
        self.data_root = os.path.join(data_root, "cifar-fs-cache-%s.pkl")
        self.split = split
        with open(self.data_root % self.split_paths[split], "rb") as infile:
            data = pkl.load(infile)

        if "image_data" in data and "class_dict" in data:
            self.features = data["image_data"]
            label_names = list(data["class_dict"].keys())
            labels = np.zeros((self.features.shape[0],), dtype=int)
            for i, name in enumerate(sorted(label_names)):
                labels[np.array(data["class_dict"][name])] = i
        elif "data" in data and "labels" in data:
            self.features = np.array(data["data"])
            labels = np.array(data["labels"])
        else:
            raise KeyError(
                "Unsupported PKL schema. Expect ('image_data','class_dict') or ('data','labels')."
            )

        del data
        super().__init__(labels, sampler, size, transforms)

    def sample_images(self, indices):
        return self.features[indices]

    def __iter__(self):
        return super().__iter__()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    # from src.tools.plot_episode import plot_episode
    sampler = FewShotSampler(5, 5, 15, 0)
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.ToTensor(),
    ])

    # Choose the backend you have on disk:
    # dataset = EpisodicCIFARFS('./cifarfs', 'train', sampler, 1000, transforms)
    dataset = EpisodicCIFARFSPkl('./cifarfs', 'train', sampler, 1000, transforms)

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    for batch in loader:
        # Expect shape (n_way * (n_shot + n_query), H, W, C) after internal sampling
        print(np.unique(batch[0]["targets"].view(20, 5).numpy()))
        # plot_episode(batch[0], classes_first=False)
        break
