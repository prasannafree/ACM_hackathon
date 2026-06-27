#!/usr/bin/env python3
"""
Minimal training script for phase_1 models.

Usage:
    python train.py --model AlexNet --datasets DATASET_1
    python train.py --model VGG16 --datasets DATASET_1 DATASET_2
    python train.py --model ResNet101 --datasets DATASET_2 --epochs 5 --batch-size 32
"""

import argparse
import csv
import importlib
import json
import os
import sys
import time

import yaml
import torch
from torch.utils.data import ConcatDataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

AVAILABLE_MODELS = ["AlexNet", "VGG16", "ResNet101"]
AVAILABLE_DATASETS = ["DATASET_1", "DATASET_2"]


def load_model_config(model_name):
    cfg_path = os.path.join(MODELS_DIR, model_name, "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_model(model_name, config, device):
    """Dynamically import and instantiate the model class."""
    model_module = importlib.import_module(f"models.{model_name}.model")
    model_class_name = config["model_details"]["model_class"]
    model_cls = getattr(model_module, model_class_name)
    model_args = config["default_training_config"].get("model_args", {})
    model = model_cls(device=str(device), args=model_args)
    return model.to(device)


def build_loaders(dataset_names, config, batch_size):
    """Build train/val loaders — combine datasets if multiple are given."""
    loader_args = config["default_training_config"].get("custom_loader_args", {})

    # Import loader from the model directory (they're all identical)
    loader_module = importlib.import_module("models.AlexNet.loader")
    build_transforms = loader_module.build_transforms
    ImageClassificationFolder = loader_module.ImageClassificationFolder

    img_size = loader_args.get("img_size", 224)
    val_frac = loader_args.get("val_frac", 0.05)
    num_workers = loader_args.get("num_workers", 4)
    seed = loader_args.get("seed", 42)

    train_tf = build_transforms(img_size=img_size, train=True)
    eval_tf = build_transforms(img_size=img_size, train=False)

    train_datasets, eval_datasets = [], []
    for ds_name in dataset_names:
        ds_path = os.path.join(DATA_DIR, ds_name)
        train_datasets.append(ImageClassificationFolder(ds_path, transform=train_tf))
        eval_datasets.append(ImageClassificationFolder(ds_path, transform=eval_tf))

    # Single or combined dataset
    if len(train_datasets) == 1:
        train_full, eval_full = train_datasets[0], eval_datasets[0]
    else:
        train_full = ConcatDataset(train_datasets)
        eval_full = ConcatDataset(eval_datasets)

    # Split into train / val
    import numpy as np
    from math import floor

    n = len(train_full)
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = floor((1.0 - val_frac) * n)
    train_idx = indices[:split].tolist()
    val_idx = indices[split:].tolist()

    train_subset = torch.utils.data.Subset(train_full, train_idx)
    val_subset = torch.utils.data.Subset(eval_full, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"[DATA] datasets={dataset_names}  total={n}  "
          f"train={len(train_subset)}  val={len(val_subset)}  "
          f"img_size={img_size}")
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description="Train a model on specified datasets")
    parser.add_argument("--model", required=True, choices=AVAILABLE_MODELS,
                        help="Model to train: AlexNet, VGG16, or ResNet101")
    parser.add_argument("--datasets", required=True, nargs="+", choices=AVAILABLE_DATASETS,
                        help="Datasets to use: DATASET_1 and/or DATASET_2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--save-model", action="store_true", help="Save trained model weights")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Model:    {args.model}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  Device:   {device}")
    print(f"{'='*60}\n")

    # ── Config ──
    config = load_model_config(args.model)
    trainer_args = config["default_training_config"].get("custom_trainer_args", {})
    if args.lr is not None:
        trainer_args["lr"] = args.lr

    # ── Model ──
    model = build_model(args.model, config, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] {args.model}  params={n_params/1e6:.2f}M\n")

    # ── Data ──
    train_loader, val_loader = build_loaders(args.datasets, config, args.batch_size)

    # ── Train ──
    trainer_module = importlib.import_module(f"models.{args.model}.trainer")
    trainer = trainer_module.CustomModelTrainer()

    print(f"\n[TRAIN] Starting {args.epochs} epochs...")
    start = time.time()

    train_results = trainer.train_model(
        model=model,
        results={},
        train_loader=train_loader,
        epochs=args.epochs,
        device=device,
        test_loader=val_loader,
        args=trainer_args,
        start_time=start,
    )

    train_time = time.time() - start
    print(f"\n[TRAIN] Done in {train_time:.1f}s")
    print(f"  Train Loss:     {train_results['loss']:.4f}")
    print(f"  Train Accuracy: {train_results['accuracy']:.2f}%")
    print(f"  Epochs:         {train_results['num_epochs']:.2f}")
    print(f"  Mini-batches:   {train_results['total_mini_batches']}")

    # ── Validate ──
    print(f"\n[VAL] Evaluating on validation set...")
    val_results = trainer.validate_model(
        model=model,
        dataloader=val_loader,
        device=str(device),
        args={},
    )

    print(f"  Val Loss:       {val_results['loss']:.4f}")
    print(f"  Val Accuracy:   {val_results['accuracy']:.2f}%")

    # ── Summary ──
    results = {
        "model": args.model,
        "datasets": args.datasets,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "device": str(device),
        "params_M": round(n_params / 1e6, 2),
        "train_loss": round(train_results["loss"], 4),
        "train_accuracy": round(train_results["accuracy"], 2),
        "val_loss": round(val_results["loss"], 4),
        "val_accuracy": round(val_results["accuracy"], 2),
        "train_time_s": round(train_time, 1),
    }

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for k, v in results.items():
        print(f"  {k:20s}: {v}")
    print(f"{'='*60}\n")

    # ── Append to CSV ──
    csv_file = os.path.join(BASE_DIR, "results.csv")
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        
        # Convert list to string for CSV
        csv_results = results.copy()
        csv_results["datasets"] = "+".join(csv_results["datasets"])
        writer.writerow(csv_results)
        
    print(f"[SAVE] Run metrics appended to {csv_file}")

    # ── Save Model Weights ──
    if args.save_model:
        out_dir = os.path.join(BASE_DIR, "outputs")
        os.makedirs(out_dir, exist_ok=True)
        ds_tag = "_".join(args.datasets)
        model_path = os.path.join(out_dir, f"{args.model}_{ds_tag}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"[SAVE] Model saved to {model_path}")

        results_path = os.path.join(out_dir, f"{args.model}_{ds_tag}_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[SAVE] Results saved to {results_path}")


if __name__ == "__main__":
    main()
