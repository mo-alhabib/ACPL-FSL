"""
Few-Shot Parallel: trains a model as a series of tasks computed in parallel on multiple GPUs.

This finetuning uses two heads:
  (1) the usual episodic head (label propagation / prototypical / linear),
  (2) our ACPL reconstruction head whose logits are refined EVERY STEP by our SAPS+EBSS
      ensemble clustering. The final training loss blends the two heads.

Goal: keep the strong behavior while ensuring our SAPS+EBSS contributes positively and stably.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from tqdm import tqdm
from haven import haven_utils as haven
from src.tools.meters import BasicMeter
from src.modules.distances import prototype_distance
from .base_wrapper import BaseWrapper
from feature_propagation import EmbeddingPropagation, LabelPropagation
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Our logits head (ACPL + SAPS/EBSS refinement)
from src.models.base_ssl.selection_methods.acpl_fsl import acpl_fsl_logit


class FinetuneWrapper(BaseWrapper):
    """Finetunes a model using an episodic scheme on multiple GPUs."""

    def __init__(self, model, nclasses, exp_dict):
        super().__init__()
        self.model = model
        self.exp_dict = exp_dict
        self.ngpu = int(self.exp_dict.get("ngpu", 1))
        self.nclasses = int(nclasses)

        key = "unlabeled_size_test"
        print("============> {} = {}".format(key, self.exp_dict.get(key, "N/A")))

        self.feature_propagation = EmbeddingPropagation()
        self.label_propagation = LabelPropagation()

        # Classifiers
        self.model.add_classifier(self.nclasses, modalities=0)
        if float(self.exp_dict.get("rotation_weight", 0.0)) > 0:
            self.model.add_classifier(4, "classifier_rot")

        # Load best pretraining checkpoint
        best_accuracy = -1
        if self.exp_dict.get("pretrained_weights_root") is not None:
            for exp_hash in os.listdir(self.exp_dict["pretrained_weights_root"]):
                base_path = os.path.join(self.exp_dict["pretrained_weights_root"], exp_hash)
                exp_dict_path = os.path.join(base_path, "exp_dict.json")
                if not os.path.exists(exp_dict_path):
                    continue
                loaded_exp_dict = haven.load_json(exp_dict_path)
                pkl_path = os.path.join(base_path, "score_list_best.pkl")
                if (
                    loaded_exp_dict.get("model", {}).get("name") == "pretraining"
                    and loaded_exp_dict.get("dataset_train", "x").split("_")[-1]
                    == self.exp_dict.get("dataset_train", "x").split("_")[-1]
                    and loaded_exp_dict.get("model", {}).get("backbone")
                    == self.exp_dict.get("model", {}).get("backbone")
                    and os.path.exists(pkl_path)
                ):
                    accuracy = haven.load_pkl(pkl_path)[-1]["val_accuracy"]
                    try:
                        self.model.load_state_dict(
                            torch.load(os.path.join(base_path, "checkpoint_best.pth"))["model"],
                            strict=False,
                        )
                        if accuracy > best_accuracy:
                            best_path = os.path.join(base_path, "checkpoint_best.pth")
                            best_accuracy = accuracy
                    except Exception:
                        continue
            assert best_accuracy > 0.1
            print("Finetuning %s with original accuracy : %f" % (base_path, best_accuracy))
            self.model.load_state_dict(torch.load(best_path)["model"], strict=False)

        # Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=float(self.exp_dict.get("lr", 0.05)),
            momentum=0.9,
            weight_decay=float(self.exp_dict.get("weight_decay", 5e-4)),
            nesterov=True,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min" if "loss" in str(self.exp_dict.get("target_loss", "val_accuracy")).lower() else "max",
            patience=int(self.exp_dict.get("patience", 10)),
        )

        # AMP + grad clip knobs
        # Default AMP OFF to match our best LP numerics unless explicitly enabled.
        self.use_amp = bool(self.exp_dict.get("amp", False))
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip = float(self.exp_dict.get("grad_clip_norm", 0.0))  # 0 => off

        # L2-normalization: OFF by default for labelprop (restores our best behavior).
        self.use_l2norm = bool(self.exp_dict.get("l2_normalize_embeddings", False))

        self.model.cuda()
        if self.ngpu > 1:
            self.parallel_model = torch.nn.DataParallel(self.model)

        # ACPL hyperparams (kept close to our best values)
        self.acpl_n_loops = int(self.exp_dict.get("acpl_n_loops", 10))
        self.acpl_lambda = float(self.exp_dict.get("acpl_lambda", 0.01))
        self.acpl_tau = float(self.exp_dict.get("acpl_tau", 0.10))

        # Loss weights
        self.cls_w = float(self.exp_dict.get("classification_weight", 0.0))
        self.acpl_w = float(self.exp_dict.get("acpl_weight", 0.8))
        self.fs_w = float(self.exp_dict.get("few_shot_weight", 1.0))

    @staticmethod
    def _l2n(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        return F.normalize(x, p=2, dim=1, eps=eps)

    # ----------------------------
    # Baseline episodic head logits
    # ----------------------------
    def get_logits(self, embeddings, support_size, query_size, nclasses):
        # Guard dtype for propagation ops
        orig_dtype = embeddings.dtype

        # Choose distance head (support legacy key "distance_tpe")
        dist_type = str(self.exp_dict.get("distance_type",
                           self.exp_dict.get("distance_tpe", "")))

        # Only L2-normalize for prototypical (NOT for label propagation)
        if self.use_l2norm and dist_type == "prototypical":
            embeddings = self._l2n(embeddings)

        # Optional embedding propagation — do it in fp32, cast back
        if bool(self.exp_dict.get("embedding_prop", False)):
            with torch.cuda.amp.autocast(False):
                embeddings = self.feature_propagation(embeddings.float())
            embeddings = embeddings.to(orig_dtype)

        b, c = embeddings.size()

        if dist_type == "labelprop":
            support_labels = (
                torch.arange(nclasses, device=embeddings.device)
                .view(1, nclasses)
                .repeat(support_size, 1)
                .view(support_size, nclasses)
            )
            unlabeled_labels = nclasses * torch.ones(
                query_size * nclasses, dtype=support_labels.dtype, device=support_labels.device
            ).view(query_size, nclasses)
            labels = torch.cat([support_labels, unlabeled_labels], 0).view(-1)

            
            with torch.cuda.amp.autocast(False):
                logits = self.label_propagation(embeddings.float(), labels, nclasses)
            logits = logits.to(orig_dtype)
            logits = logits.view(-1, nclasses, nclasses)[
                support_size:(support_size + query_size), ...
            ].view(-1, nclasses)

        elif dist_type == "prototypical":
            embeddings = embeddings.view(-1, nclasses, c)
            support_embeddings = embeddings[:support_size]
            query_embeddings = embeddings[support_size:]
            logits = prototype_distance(
                (support_embeddings.view(1, support_size, nclasses, c), False),
                (query_embeddings.view(1, query_size, nclasses, c), False),
            ).view(query_size * nclasses, nclasses)

        else:
            # default linear
            logits = self.model.classifier(embeddings).view(-1, nclasses)

        return logits

    # ----------------------------
    # Train step
    # ----------------------------
    def train_on_batch(self, batch):
        """Computes the loss on an episode."""
        LABEL_SMOOTH = float(self.exp_dict.get("label_smoothing", 0.0))

        episode = batch[0]
        nclasses = int(episode["nclasses"])
        support_size = int(episode["support_size"])
        query_size = int(episode["query_size"])

        labels = (
            episode["targets"][: (support_size + query_size) * nclasses, ...]
            .view(support_size + query_size, nclasses, -1)
            .cuda(non_blocking=True)
            .long()
        )

        c = int(episode["channels"])
        h = int(episode["height"])
        w = int(episode["width"])

        tx = episode["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = episode["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)
        augmented_tx = (
            episode["augmented_set"][:support_size, ...]
            .view(support_size, nclasses, c, h, w)
            .cuda(non_blocking=True)
        )

        x = torch.cat([augmented_tx, vx], 0).view(-1, c, h, w).cuda(non_blocking=True)

        with autocast(enabled=self.use_amp):
            # Backbone forward
            if self.ngpu > 1:
                embeddings_all = self.parallel_model(x, is_support=True)
            else:
                embeddings_all = self.model(x, is_support=True)

            # Optional embedding propagation in fp32
            if bool(self.exp_dict.get("embedding_prop", False)):
                with torch.cuda.amp.autocast(False):
                    embeddings_all = self.feature_propagation(embeddings_all.float())
                embeddings_all = embeddings_all.to(x.dtype)

            embeddings = embeddings_all[: (support_size + query_size) * nclasses, ...]
            a, b = embeddings.size()

            # (1) baseline episodic head
            student_logits = self.get_logits(embeddings, support_size, query_size, nclasses)

            # (2) ACPL + SAPS/EBSS teacher (fp32 block: avoids Half solvers)
            emb_s = embeddings_all[: support_size * nclasses, ...]
            emb_q = embeddings_all[support_size * nclasses : (support_size + query_size) * nclasses, ...]
            support_labels = torch.arange(nclasses, device=emb_s.device).repeat(support_size)

            with torch.cuda.amp.autocast(enabled=False):
                teacher_probs, _ = acpl_fsl_logit(
                    emb_s,
                    emb_q,
                    support_labels,
                    nclasses,
                    n_loops=int(self.exp_dict.get("acpl_n_loops", self.acpl_n_loops)),
                    lam=float(self.exp_dict.get("acpl_lambda", self.acpl_lambda)),
                    tau=float(self.exp_dict.get("acpl_tau", self.acpl_tau)),
                    # refinement knobs (use our defaults if provided)
                    alpha_min=float(self.exp_dict.get("logit_alpha_min", 0.02)),
                    alpha_max=float(self.exp_dict.get("logit_alpha_max", 0.08)),
                    delta_guard=float(self.exp_dict.get("logit_delta_guard", 0.05)),
                    dbscan_eps=float(self.exp_dict.get("acpl_dbscan_eps", 0.6)),
                    dbscan_min_samples=int(self.exp_dict.get("acpl_dbscan_min_samples", 5)),
                    knn_k=int(self.exp_dict.get("acpl_knn_k", 5)),
                    graph_k=int(self.exp_dict.get("acpl_graph_k", 10)),
                    km_random_state=int(self.exp_dict.get("acpl_km_random_state", 1996)),
                    kmeans_iter=int(self.exp_dict.get("acpl_kmeans_iter", 20)),
                    knng_mode=str(self.exp_dict.get("acpl_knng_builder", "cdp")),
                    cdp_alpha=float(self.exp_dict.get("acpl_cdp_alpha", 0.5)),
                    use_faiss_kmeans=bool(self.exp_dict.get("acpl_use_faiss_kmeans", True)),
                    use_faiss_knn=bool(self.exp_dict.get("acpl_use_faiss_knn", True)),
                    standardize_for_dbscan=bool(self.exp_dict.get("acpl_standardize_dbscan", True)),
                )

            # ---------- losses ----------
            loss = torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)

            # supervised classifier aux loss
            if self.cls_w > 0:
                aux_logits = self.model.classifier(embeddings.view(a, b))
                loss = loss + self.cls_w * F.cross_entropy(
                    aux_logits, labels.view(-1), label_smoothing=LABEL_SMOOTH
                )

            # ground-truth query labels (episodic structure)
            query_labels = (
                torch.arange(nclasses, device=embeddings.device)
                .view(1, nclasses)
                .repeat(query_size, 1)
                .view(-1)
            )

            # blend teacher + baseline (our validated recipe)
            # teacher_probs are probabilities; this matched our previous best.
            loss_teacher = F.cross_entropy(teacher_probs, query_labels)
            loss_student = F.cross_entropy(student_logits, query_labels)
            loss = loss + self.fs_w * (self.acpl_w * loss_teacher + (1.0 - self.acpl_w) * loss_student)

        # backward (AMP + grad clip handled by the outer loop)
        self.scaler.scale(loss / float(self.exp_dict.get("tasks_per_batch", 1))).backward()
        if self.grad_clip > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        return loss

    # ----------------------------
    # Predict (validation/test)
    # ----------------------------
    def predict_on_batch(self, batch):
        """Computes the baseline logits of an episode (evaluation head unchanged)."""
        nclasses = int(batch["nclasses"])
        support_size = int(batch["support_size"])
        query_size = int(batch["query_size"])

        c = int(batch["channels"])
        h = int(batch["height"])
        w = int(batch["width"])

        tx = batch["support_set"].view(support_size, nclasses, c, h, w).cuda(non_blocking=True)
        vx = batch["query_set"].view(query_size, nclasses, c, h, w).cuda(non_blocking=True)

        x = torch.cat([tx, vx], 0).view(-1, c, h, w).cuda(non_blocking=True)
        with torch.no_grad(), autocast(enabled=self.use_amp):
            if self.ngpu > 1:
                embeddings = self.parallel_model(x, is_support=True)
            else:
                embeddings = self.model(x, is_support=True)
        return self.get_logits(embeddings, support_size, query_size, nclasses)

    @torch.no_grad()
    def val_on_batch(self, batch):
        """Loss and accuracy for validation (unchanged evaluation head)."""
        nclasses = int(batch["nclasses"])
        query_size = int(batch["query_size"])

        logits = self.predict_on_batch(batch)
        query_labels = (
            torch.arange(nclasses, device=logits.device).view(1, nclasses).repeat(query_size, 1).view(-1)
        )
        loss = F.cross_entropy(logits, query_labels)
        accuracy = float(logits.max(-1)[1].eq(query_labels).float().mean())
        return loss, accuracy

    # ----------------------------
    # Train / Val / Test loops
    # ----------------------------
    def train_on_loader(self, data_loader, max_iter=None, debug_plot_path=None):
        self.model.train()
        train_loss_meter = BasicMeter.get("train_loss").reset()
        self.optimizer.zero_grad()

        tpb = int(self.exp_dict.get("tasks_per_batch", 1))

        for batch_idx, batch in enumerate(tqdm(data_loader, ncols=80, ascii="->")):
            loss = self.train_on_batch(batch)
            train_loss_meter.update(float(loss), 1)

            # step optimizer every tasks_per_batch
            if ((batch_idx + 1) % tpb) == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            if (max_iter is not None) and (batch_idx + 1 == max_iter):
                break

        return {"train_loss": train_loss_meter.mean()}

    @torch.no_grad()
    def val_on_loader(self, data_loader, max_iter=None):
        self.model.eval()
        val_loss_meter = BasicMeter.get("val_loss").reset()
        val_accuracy_meter = BasicMeter.get("val_accuracy").reset()

        for batch_idx, _data in enumerate(tqdm(data_loader, ncols=80, ascii="->")):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            val_loss_meter.update(float(loss), 1)
            val_accuracy_meter.update(float(accuracy), 1)

        target = str(self.exp_dict.get("target_loss", "val_accuracy"))
        loss = BasicMeter.get(target, recursive=True, force=False).mean()
        self.scheduler.step(loss)  # update LR scheduler
        return {"val_loss": val_loss_meter.mean(), "val_accuracy": val_accuracy_meter.mean()}

    @torch.no_grad()
    def test_on_loader(self, data_loader, max_iter=None):
        self.model.eval()
        test_loss_meter = BasicMeter.get("test_loss").reset()
        test_accuracy_meter = BasicMeter.get("test_accuracy").reset()
        test_accuracy = []

        for batch_idx, _data in enumerate(data_loader):
            batch = _data[0]
            loss, accuracy = self.val_on_batch(batch)
            test_loss_meter.update(float(loss), 1)
            test_accuracy_meter.update(float(accuracy), 1)
            test_accuracy.append(float(accuracy))

        from scipy.stats import sem, t

        confidence = 0.95
        n = max(1, len(test_accuracy))
        std_err = sem(np.array(test_accuracy)) if n > 1 else 0.0
        h = std_err * t.ppf((1 + confidence) / 2, n - 1) if n > 1 else 0.0
        return {
            "test_loss": test_loss_meter.mean(),
            "test_accuracy": test_accuracy_meter.mean(),
            "test_confidence": h,
        }

    # ----------------------------
    # Utilities
    # ----------------------------
    def get_state_dict(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.model.load_state_dict(state_dict["model"])
        self.scheduler.load_state_dict(state_dict["scheduler"])

    def get_lr(self):
        ret = {}
        for i, pg in enumerate(self.optimizer.param_groups):
            ret["current_lr_%d" % i] = float(pg["lr"])
        return ret

    def is_end_of_training(self):
        lr = self.get_lr()["current_lr_0"]
        return lr <= (float(self.exp_dict.get("lr", 0.05)) * float(self.exp_dict.get("min_lr_decay", 0.01)))
