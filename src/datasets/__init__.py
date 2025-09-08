# from . import trancos, fish_reg
from torchvision import transforms
import torchvision
import os
from src import utils as ut
import pandas as pd
import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100
from .episodic_dataset import FewShotSampler
from .episodic_miniimagenet import EpisodicMiniImagenet, EpisodicMiniImagenetPkl
from .miniimagenet import NonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenet, RotatedNonEpisodicMiniImagenetPkl
from .episodic_tiered_imagenet import EpisodicTieredImagenet
from .tiered_imagenet import RotatedNonEpisodicTieredImagenet, NonEpisodicTieredImagenet
from .cub import RotatedNonEpisodicCUB, NonEpisodicCUB
from .episodic_cub import EpisodicCUB

# ---- CDFSL (ChestX, CropDiseases, EuroSAT, ISIC) ----
try:
    from .cdfsl.episodic_cdfsl import EpisodicCDFSL
    _HAS_CDFSL = True
except Exception:
    EpisodicCDFSL = None
    _HAS_CDFSL = False


def get_dataset(dataset_name,
                data_root,
                split,
                transform,
                classes,
                support_size,
                query_size,
                unlabeled_size,
                n_iters):

    # ---------- CDFSL hook (added) ----------
    # Accept names like:
    #   episodic_cdfsl_chestx
    #   episodic_cdfsl_cropdiseases
    #   episodic_cdfsl_eurosat
    #   episodic_cdfsl_isic
    lower_name = str(dataset_name).lower()
    if lower_name.startswith("episodic_cdfsl_"):
        if not _HAS_CDFSL:
            raise RuntimeError(
                "CDFSL dataset requested but 'EpisodicCDFSL' is not available. "
                "Ensure src/datasets/cdfsl/episodic_cdfsl.py exists and imports correctly."
            )
        domain = _infer_cdfsl_domain(lower_name)
        tf = get_cdfsl_transformer(transform, split, domain)
        dataset = EpisodicCDFSL(
            data_root=data_root,
            split=split,
            transforms=tf,           # PIL-friendly transforms
            classes=classes,
            support_size=support_size,
            query_size=query_size,
            n_iters=n_iters,
            unlabeled_size=unlabeled_size,
            domain=domain
        )
        return dataset
    # ----------------------------------------

    transform_func = get_transformer(transform, split)

    if dataset_name == "rotated_miniimagenet":
        dataset = RotatedNonEpisodicMiniImagenet(data_root,
                                                 split,
                                                 transform_func)

    elif dataset_name == "miniimagenet":
        dataset = NonEpisodicMiniImagenet(data_root,
                                          split,
                                          transform_func)

    elif dataset_name == "episodic_miniimagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenet(data_root=data_root,
                                       split=split,
                                       sampler=few_shot_sampler,
                                       size=n_iters,
                                       transforms=transform_func)

    elif dataset_name == "rotated_episodic_miniimagenet_pkl":
        dataset = RotatedNonEpisodicMiniImagenetPkl(data_root=data_root,
                                                    split=split,
                                                    transforms=transform_func)

    elif dataset_name == "episodic_miniimagenet_pkl":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicMiniImagenetPkl(data_root=data_root,
                                          split=split,
                                          sampler=few_shot_sampler,
                                          size=n_iters,
                                          transforms=transform_func)

    elif dataset_name == "cub":
        dataset = NonEpisodicCUB(data_root, split, transform_func)

    elif dataset_name == "rotated_cub":
        dataset = RotatedNonEpisodicCUB(data_root, split, transform_func)

    elif dataset_name == "episodic_cub":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicCUB(data_root=data_root,
                              split=split,
                              sampler=few_shot_sampler,
                              size=n_iters,
                              transforms=transform_func)

    elif dataset_name == "tiered-imagenet":
        dataset = NonEpisodicTieredImagenet(data_root, split, transform_func)

    elif dataset_name == "rotated_tiered-imagenet":
        dataset = RotatedNonEpisodicTieredImagenet(data_root, split, transform_func)

    elif dataset_name == "episodic_tiered-imagenet":
        few_shot_sampler = FewShotSampler(classes, support_size, query_size, unlabeled_size)
        dataset = EpisodicTieredImagenet(data_root,
                                         split=split,
                                         sampler=few_shot_sampler,
                                         size=n_iters,
                                         transforms=transform_func)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset


# ===================================================
# helpers

def _infer_cdfsl_domain(name: str) -> str:
    if "chestx" in name:
        return "chestx"
    if "crop" in name or "cropdiseases" in name:
        return "cropdiseases"
    if "eurosat" in name:
        return "eurosat"
    if "isic" in name:
        return "isic"
    return "generic"


def get_cdfsl_transformer(transform_name: str, split: str, domain: str):
    """
    PIL-friendly transforms for CDFSL (images are opened as PIL in EpisodicCDFSL).
    Mirrors WRN pipelines but without ToPILImage().
    Adds Grayscale->RGB handling for medical domains when needed.
    """
    # Common base
    def _base_val():
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((92, 92)),
            torchvision.transforms.CenterCrop(80),
            torchvision.transforms.ToTensor()
        ])

    def _base_train():
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((92, 92)),
            torchvision.transforms.CenterCrop(80),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])

    # Domain guard (ChestX/ISIC may be grayscale) -> force 3ch
    def _prepend_grayscale_3(tf):
        return torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            *tf.transforms  # type: ignore[attr-defined]
        ])

    # Map names to pipelines 
    if transform_name == "wrn_finetune_train":
        tf = _base_train()
    elif transform_name == "wrn_pretrain_train":
        tf = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1.0)),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.ToTensor()
        ])
    else:
        # covers "wrn_val" and any other "wrn_*" eval transforms
        tf = _base_val()

    if domain in {"chestx", "isic"}:
        tf = _prepend_grayscale_3(tf)

    return tf


def get_transformer(transform, split):
    if transform == "data_augmentation":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((84, 84)),
            torchvision.transforms.ToTensor()
        ])
        return transform

    if "{}_{}".format(transform, split) == "cifar_train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if "{}_{}".format(transform, split) in ("cifar_test", "cifar_val"):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                             (0.2023, 0.1994, 0.2010)),
        ])
        return transform

    if transform == "basic":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((84, 84)),
            torchvision.transforms.ToTensor()
        ])
        return transform

    if transform == "wrn_pretrain_train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomResizedCrop((80, 80), scale=(0.08, 1)),
            torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            torchvision.transforms.ToTensor()
        ])
        return transform

    elif transform == "wrn_finetune_train":
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((92, 92)),
            torchvision.transforms.CenterCrop(80),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor()
        ])
        return transform

    elif "wrn" in transform:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((92, 92)),
            torchvision.transforms.CenterCrop(80),
            torchvision.transforms.ToTensor()
        ])
        return transform

    raise NotImplementedError
