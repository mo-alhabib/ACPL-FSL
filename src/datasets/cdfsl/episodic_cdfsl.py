import os, random, torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

class EpisodicCDFSL(torch.utils.data.Dataset):
    """
    Generic CDFSL episodic loader (ChestX, CropDiseases, EuroSAT, ISIC).
    Assumes folder-per-class. Builds episodes on-the-fly.
    Handles domain quirks:
      - ChestX, ISIC: robust to grayscale -> expands to 3ch
      - EuroSAT: accepts .tif/.tiff as well as common image formats
    """
    def __init__(self, *,
                 data_root: str,
                 split: str,
                 transform: str,
                 classes: int,
                 support_size: int,
                 query_size: int,
                 n_iters: int,
                 unlabeled_size: int = 0,
                 domain: str = "generic",
                 seed: int = 1996):
        super().__init__()
        assert split in {"train", "val", "test"}
        self.root = data_root
        self.n_way = int(classes)
        self.k_shot = int(support_size)
        self.q_query = int(query_size)
        self.u_unlab = int(unlabeled_size)
        self.n_iters = int(n_iters)
        self.domain = str(domain).lower()
        self.rng = random.Random(seed)

        # enumerate class folders
        self.class_names = sorted([d for d in os.listdir(self.root)
                                   if os.path.isdir(os.path.join(self.root, d))])
        if len(self.class_names) == 0:
            raise RuntimeError(f"No class folders found under {self.root}")

        # gather image paths per class
        self.images_by_class = []
        for cname in self.class_names:
            cdir = os.path.join(self.root, cname)
            files = [os.path.join(cdir, f) for f in os.listdir(cdir)
                     if f.lower().endswith(_ALLOWED_EXT)]
            if len(files) > 0:
                self.images_by_class.append(files)

        if len(self.images_by_class) < self.n_way:
            raise RuntimeError(
                f"Found {len(self.images_by_class)} classes, but need n_way={self.n_way}"
            )

        self.tf = self._build_transform(transform, self.domain)

    def _build_transform(self, name: str, domain: str):
        # minimally opinionated; keep pipeline shape (Resize->CenterCrop->ToTensor)
        # domain-specific pre-step: ensure 3 channels for medical/grayscale domains.
        base = [
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
        ]
        if domain in {"chestx", "isic"}:
            # force 3-channel: either by Grayscale(3) or RGB convert in __getitem__
            # Using Grayscale(3) is safest if images are single-channel
            return transforms.Compose([transforms.Grayscale(num_output_channels=3)] + base)
        else:
            return transforms.Compose(base)

    def __len__(self):
        return self.n_iters

    def _open_rgb(self, fp: str):
        # robust loader: some EuroSAT images are .tif; some ChestX/ISIC are grayscale.
        img = Image.open(fp)
        if img.mode not in ("RGB", "RGBA"):
            # convert all to RGB (Grayscale->RGB etc.)
            img = img.convert("RGB")
        elif img.mode == "RGBA":
            img = img.convert("RGB")
        return img

    def __getitem__(self, idx):
        # sample classes for the episode
        cls_idx = self.rng.sample(range(len(self.images_by_class)), self.n_way)

        support_imgs, query_imgs, targets = [], [], []
        for ci, c in enumerate(cls_idx):
            files = list(self.images_by_class[c])
            self.rng.shuffle(files)
            need = self.k_shot + self.q_query
            if len(files) < need:
                reps = (need + len(files) - 1) // len(files)
                files = (files * reps)[:need]

            s_files = files[:self.k_shot]
            q_files = files[self.k_shot:self.k_shot + self.q_query]

            for fp in s_files:
                img = self._open_rgb(fp)
                support_imgs.append(self.tf(img))
                targets.append(ci)

            for fp in q_files:
                img = self._open_rgb(fp)
                query_imgs.append(self.tf(img))
                targets.append(ci)

        support = torch.stack(support_imgs, dim=0)  # (Ns, C, H, W)
        query   = torch.stack(query_imgs, dim=0)    # (Nq, C, H, W)

        episode = {
            "nclasses": self.n_way,
            "support_size": self.k_shot,
            "query_size": self.q_query,
            "unlabeled_size": 0,
            "channels": support.size(1),
            "height": support.size(2),
            "width": support.size(3),
            "support_set": support.view(self.k_shot, self.n_way, *support.shape[1:]),
            "query_set":   query.view(self.q_query, self.n_way, *query.shape[1:]),
            # use support as "augmented_set" stub to match existing finetune API
            "augmented_set": support.view(self.k_shot, self.n_way, *support.shape[1:]),
            "targets": torch.tensor(targets, dtype=torch.long)
        }
        return episode, {}
