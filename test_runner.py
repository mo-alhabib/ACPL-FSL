import os
import json
import pickle
import pprint
import random
import argparse
import numpy as np
import torch

from torch.utils.data import DataLoader  
from haven import haven_utils as hu

from src import datasets, models
from src.models import backbones


# ---------------------------
# Small utilities 
# ---------------------------

def _exp_dir(savedir_base, title):
    """Keep the same convention: use the provided title as experiment id."""
    exp_id = title
    savedir = os.path.join(savedir_base, exp_id)
    os.makedirs(savedir, exist_ok=True)
    return savedir


def _save_and_show_cfg(savedir, exp_dict):
    """Persist config and pretty-print (unchanged side effects)."""
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    pprint.pprint(exp_dict)
    print(f"Experiment saved in {savedir}")


def _build_test_set(exp_dict, datadir):
    """Instantiate test split exactly as configured."""
    return datasets.get_dataset(
        dataset_name=exp_dict["dataset_test"],
        data_root=os.path.join(datadir, exp_dict["dataset_test_root"]),
        split="test",
        transform=exp_dict["transform_val"],
        classes=exp_dict["classes_test"],
        support_size=exp_dict["support_size_test"],
        query_size=exp_dict["query_size_test"],
        n_iters=exp_dict["test_iters"],
        unlabeled_size=exp_dict["unlabeled_size_test"],
    )


def _build_loader(test_set, num_workers):
    """Test loader parity: bs=1, no shuffle, drop_last=False, identity collate."""
    return torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=False,
    )


def _build_model(exp_dict, savedir_base):
    """Backbone + wrapper; """
    backbone = backbones.get_backbone(
        backbone_name=exp_dict["model"]["backbone"],
        exp_dict=exp_dict,
    )
    model = models.get_model(
        model_name=exp_dict["model"]["name"],
        backbone=backbone,
        n_classes=exp_dict["n_classes"],
        exp_dict=exp_dict,
        pretrained_weights_dir="just some stupid path",  # preserved
        savedir_base=None,                               # preserved
        load_pretrained=False,                           # preserved
    )
    return model


def _load_ckpt_if_any(model, ckpt):
    """Strict load as in original."""
    if ckpt is not None:
        print(f"=> Model from `{ckpt}` loaded")
        state = torch.load(ckpt, map_location="cpu")
        model.model.load_state_dict(state["model"], strict=True)


def _cached_results_path(savedir):
    return os.path.join(savedir, "score_list.pkl")


def _run_or_load_test(model, test_loader, savedir):
    """Run model.test_on_loader once and cache results, or load cached results."""
    cache_path = _cached_results_path(savedir)
    if not os.path.exists(cache_path):
        #  (max_iter=None)
        test_dict = model.test_on_loader(test_loader, max_iter=None)
        hu.save_pkl(cache_path, [test_dict])
    else:
        print("=> Cached result loaded")

    with open(cache_path, "rb") as f:
        test_dict = pickle.load(f)
    return test_dict


# ---------------------------
# Public API 
# ---------------------------

def trainval(exp_dict, savedir_base, datadir, ckpt, title=None, num_workers=0):
    """Runs test-time evaluation with result caching. Signature unchanged."""
    # workspace & config
    savedir = _exp_dir(savedir_base, title)
    _save_and_show_cfg(savedir, exp_dict)

    # data
    test_set = _build_test_set(exp_dict, datadir)
    test_loader = _build_loader(test_set, num_workers)

    # model
    model = _build_model(exp_dict, savedir_base)
    _load_ckpt_if_any(model, ckpt)

    
    assert exp_dict["model"]["name"] == "ssl"

    # run or load cached
    test_dict = _run_or_load_test(model, test_loader, savedir)
    print("=>", test_dict)


# ---------------------------
# CLI 
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="json config path")
    parser.add_argument("ckpt", type=str, help="checkpoint path")
    parser.add_argument(
        "-sb", "--savedir_base", required=True,
        help="Testing result will be saved under {savedir_base}/[title]"
    )
    parser.add_argument("-d", "--datadir", default="data/")
    parser.add_argument("-nw", "--num_workers", default=2, type=int)
    parser.add_argument(
        "-s", "--selection", default="ACPL",
        help="Pseudo-label generation method, choose from {ssl, ACPL}"
    )
    parser.add_argument("--seed", default=1996, type=int)
    parser.add_argument("-t", "--title", default=None, type=str)
    args = parser.parse_args()

    # deterministic seeds 
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open(args.cfg) as f:
        exp_dict = json.load(f)

    exp_dict["selection"] = args.selection
    exp_dict["seed"] = args.seed

    trainval(
        exp_dict=exp_dict,
        savedir_base=args.savedir_base,
        datadir=args.datadir,
        ckpt=args.ckpt,
        title=args.title,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
