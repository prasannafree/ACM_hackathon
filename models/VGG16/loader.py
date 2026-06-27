import os
from math import floor

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def resolve_dataset_root(path):
    path = os.path.abspath(path)
    if os.path.isdir(os.path.join(path, "images")) and os.path.isdir(
        os.path.join(path, "labels")
    ):
        return path

    subdirs = [
        d
        for d in sorted(os.listdir(path))
        if os.path.isdir(os.path.join(path, d))
        and os.path.isdir(os.path.join(path, d, "images"))
        and os.path.isdir(os.path.join(path, d, "labels"))
    ]
    if len(subdirs) == 1:
        return os.path.join(path, subdirs[0])
    if len(subdirs) > 1:
        raise ValueError(
            f"{path} contains multiple dataset folders {subdirs}; "
            "point the data path at a specific one."
        )
    raise ValueError(f"Could not find an images/ + labels/ dataset under {path}")


def _read_label(label_path):
    with open(label_path, "r") as f:
        line = f.readline().strip()
    return int(line.split()[0])


class ImageClassificationFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = resolve_dataset_root(root)
        self.transform = transform
        img_dir = os.path.join(self.root, "images")
        lbl_dir = os.path.join(self.root, "labels")

        self.samples = []
        for fname in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTS:
                continue
            label_path = os.path.join(lbl_dir, f"{stem}.txt")
            if os.path.exists(label_path):
                self.samples.append((os.path.join(img_dir, fname), label_path))

        if not self.samples:
            raise ValueError(f"No (image, label) pairs found under {self.root}")

        self.targets = [_read_label(lp) for _, lp in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, _ = self.samples[idx]
        target = self.targets[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(img_size=224, train=True, mean=CIFAR10_MEAN, std=CIFAR10_STD):
    if train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomCrop(img_size, padding=max(4, img_size // 8)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )


class CustomDataLoader:
    def __init__(self) -> None:
        pass

    def get_train_test_dataset_loaders(
        self, batch_size=64, dataset_path=None, args: dict = None
    ):
        args = args or {}
        img_size = args.get("img_size", 224)
        val_frac = args.get("val_frac", 0.05)
        num_workers = args.get("num_workers", 4)
        seed = args.get("seed", 42)
        pin_memory = args.get("pin_memory", True)

        eval_tf = build_transforms(img_size=img_size, train=False)

        if args.get("eval_only") or (dataset_path and "val_data" in dataset_path):
            dataset = ImageClassificationFolder(dataset_path, transform=eval_tf)
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            print(
                "CustomDataLoader.get_train_test_dataset_loaders:: eval-only, "
                f"samples={len(dataset)} batches={len(test_loader)} img_size={img_size}"
            )
            return train_loader, test_loader

        train_tf = build_transforms(img_size=img_size, train=True)
        train_view = ImageClassificationFolder(dataset_path, transform=train_tf)
        eval_view = ImageClassificationFolder(dataset_path, transform=eval_tf)

        n = len(train_view)
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
        split = floor((1.0 - val_frac) * n)
        train_idx = indices[:split].tolist()
        val_idx = indices[split:].tolist()

        train_dataset = torch.utils.data.Subset(train_view, train_idx)
        test_dataset = torch.utils.data.Subset(eval_view, val_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        print(
            "CustomDataLoader.get_train_test_dataset_loaders:: "
            f"train={len(train_dataset)} val={len(test_dataset)} "
            f"train_batches={len(train_loader)} val_batches={len(test_loader)} "
            f"img_size={img_size}"
        )
        return train_loader, test_loader
