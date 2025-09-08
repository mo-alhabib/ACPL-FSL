import os
import json
import argparse
import pprint
import pandas as pd
import torch

from torch.utils.data import DataLoader  
from haven import haven_utils as hu
from haven import haven_chk as hc

from src import utils as ut
from src import datasets, models
from src.models import backbones


# ---------------------------
# Helpers 
# ---------------------------

def _compute_exp_id(exp_dict, title):
    """Prefer an explicit title; otherwise derive a readable id from the config."""
    if title is not None:
        return str(title)
    # common practice: hashed cfg + optional name stem if present
    name_stem = str(exp_dict.get('name', 'exp')).split('.')[0]
    return f"{hu.hash_dict(exp_dict)}-{name_stem}"


def _prepare_workspace(savedir_base, exp_id, exp_dict, reset=False):
    """Create experiment directory, logger, and persist config."""
    savedir = os.path.join(savedir_base, exp_id)
    os.makedirs(savedir, exist_ok=True)

    if reset:
        hc.delete_experiment(savedir, backup_flag=True)
        os.makedirs(savedir, exist_ok=True)

    ut.setup_logger(os.path.join(savedir, "train_log.txt"))
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)

    pprint.pprint(exp_dict)
    print(f"Experiment directory: {savedir}")
    return savedir


def _build_datasets(exp_dict, datadir):
    """Instantiate train/val/test episodic datasets exactly as configured."""
    train_set = datasets.get_dataset(
        dataset_name=exp_dict["dataset_train"],
        data_root=os.path.join(datadir, exp_dict["dataset_train_root"]),
        split="train",
        transform=exp_dict["transform_train"],
        classes=exp_dict["classes_train"],
        support_size=exp_dict["support_size_train"],
        query_size=exp_dict["query_size_train"],
        n_iters=exp_dict["train_iters"],
        unlabeled_size=exp_dict["unlabeled_size_train"],
    )

    val_set = datasets.get_dataset(
        dataset_name=exp_dict["dataset_val"],
        data_root=os.path.join(datadir, exp_dict["dataset_val_root"]),
        split="val",
        transform=exp_dict["transform_val"],
        classes=exp_dict["classes_val"],
        support_size=exp_dict["support_size_val"],
        query_size=exp_dict["query_size_val"],
        n_iters=exp_dict.get("val_iters", None),
        unlabeled_size=exp_dict["unlabeled_size_val"],
    )

    test_set = datasets.get_dataset(
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
    return train_set, val_set, test_set


def _build_loaders(train_set, val_set, test_set, exp_dict, num_workers):
    """Dataloaders (unchanged semantics, same collate fns & drop_last)."""
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        collate_fn=ut.get_collate(exp_dict["collate_fn"]),
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x,
        drop_last=True,
    )
    return train_loader, val_loader, test_loader


def _build_model(exp_dict, savedir_base, ckpt):
    """Create backbone + model wrapper; optionally load a checkpoint."""
    backbone = backbones.get_backbone(
        backbone_name=exp_dict["model"]["backbone"],
        exp_dict=exp_dict,
    )
    model = models.get_model(
        model_name=exp_dict["model"]["name"],
        backbone=backbone,
        n_classes=exp_dict["n_classes"],
        exp_dict=exp_dict,
        pretrained_weights_dir=None,
        savedir_base=savedir_base,
    )

    if ckpt is not None:
        print(f"=> Loading model weights from '{ckpt}'")
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.model.load_state_dict(state["model"], strict=False)
        if missing:
            print("Missing keys:", missing)
        if unexpected:
            print("Unexpected keys:", unexpected)

    return model


def _resume_if_any(model, savedir):
    """Resume training if previous checkpoints exist; otherwise start fresh."""
    ckpt_path = os.path.join(savedir, "checkpoint.pth")
    scores_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(scores_path):
        print("=> Resuming from existing checkpoint.")
        model.load_state_dict(hu.torch_load(ckpt_path))
        score_history = hu.load_pkl(scores_path)
        start_epoch = score_history[-1]["epoch"] + 1
    else:
        print("=> Starting a new run.")
        score_history = []
        start_epoch = 0

    return ckpt_path, scores_path, score_history, start_epoch


def _is_best_so_far(score_df, target_key, current_row):
    """Replicates previous 'best' logic (max for accuracy, min otherwise)."""
    if len(score_df) <= 1:
        return True  # first row is trivially best
    if "accuracy" in target_key:
        prev_best = score_df[target_key][:-1].max()
        return current_row[target_key] >= prev_best
    else:
        prev_best = score_df[target_key][:-1].min()
        return current_row[target_key] <= prev_best


# ---------------------------
# Public API 
# ---------------------------

def trainval(exp_dict, savedir_base, datadir, reset=False, num_workers=0, title=None, ckpt=None):
    """Train + validate with checkpointing. Signature and side effects unchanged."""
    # 1) Workspace setup
    exp_id = _compute_exp_id(exp_dict, title)
    savedir = _prepare_workspace(savedir_base, exp_id, exp_dict, reset=reset)

    # 2) Data
    train_set, val_set, test_set = _build_datasets(exp_dict, datadir)
    train_loader, val_loader, test_loader = _build_loaders(train_set, val_set, test_set, exp_dict, num_workers)

    # 3) Model
    model = _build_model(exp_dict, savedir_base, ckpt)

    # 4) Resume or start fresh
    ckpt_path, scores_path, score_history, start_epoch = _resume_if_any(model, savedir)

    # 5) Loop
    max_epoch = int(exp_dict["max_epoch"])
    target_key = str(exp_dict["target_loss"])
    for epoch in range(start_epoch, max_epoch):
        row = {"epoch": epoch}
        row.update(model.get_lr())

        # train
        row.update(model.train_on_loader(train_loader))

        # validate
        row.update(model.val_on_loader(val_loader))
        
        # row.update(model.test_on_loader(test_loader))

        # bookkeeping
        score_history.append(row)
        score_df = pd.DataFrame(score_history)
        print(score_df.tail())

        # save latest
        hu.save_pkl(scores_path, score_history)
        hu.torch_save(ckpt_path, model.get_state_dict())
        print(f"Saved latest to: {savedir}")

        # save best
        if _is_best_so_far(score_df, target_key, row):
            hu.save_pkl(os.path.join(savedir, "score_list_best.pkl"), score_history)
            hu.torch_save(os.path.join(savedir, "checkpoint_best.pth"), model.get_state_dict())
            print(f"Saved BEST to: {savedir}")

        # optional early stop
        if model.is_end_of_training():
            print("Stopping early due to scheduler/min-lr condition.")
            return


# ---------------------------
# CLI 
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cfg", type=str, help="path to json config")
    parser.add_argument("--ckpt", type=str, default=None, help="resume weights (optional)")
    parser.add_argument("-sb", "--savedir_base", required=True)
    parser.add_argument("-d", "--datadir", default="data/")
    parser.add_argument("-nw", "--num_workers", type=int, default=2)
    parser.add_argument("-t", "--title", default=None, type=str)
    args = parser.parse_args()

    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

    with open(args.cfg, "r") as f:
        exp_dict = json.load(f)

    trainval(
        exp_dict=exp_dict,
        savedir_base=args.savedir_base,
        datadir=args.datadir,
        reset=False,
        num_workers=args.num_workers,
        title=args.title,
        ckpt=args.ckpt,
    )


if __name__ == "__main__":
    main()
