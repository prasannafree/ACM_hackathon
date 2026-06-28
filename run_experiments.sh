#!/bin/bash

source venv/bin/activate

echo "==============================================="
echo "STARTING OVERNIGHT TRAINING BATCH"
echo "==============================================="

echo ">>> Training AlexNet on DATASET_2..."
python train.py --model AlexNet --datasets DATASET_2 --epochs 40 --batch-size 64

echo ">>> Training VGG16 on DATASET_2..."
python train.py --model VGG16 --datasets DATASET_2 --epochs 40 --batch-size 64

echo ">>> Training ResNet101 on DATASET_2..."
python train.py --model ResNet101 --datasets DATASET_2 --epochs 40 --batch-size 64

echo ">>> Training AlexNet on DATASET_1 and DATASET_2..."
python train.py --model AlexNet --datasets DATASET_1 DATASET_2 --epochs 40 --batch-size 64

echo ">>> Training VGG16 on DATASET_1 and DATASET_2..."
python train.py --model VGG16 --datasets DATASET_1 DATASET_2 --epochs 40 --batch-size 64

echo ">>> Training ResNet101 on DATASET_1 and DATASET_2..."
python train.py --model ResNet101 --datasets DATASET_1 DATASET_2 --epochs 40 --batch-size 64

echo "==============================================="
echo "ALL TRAINING COMPLETE!"
echo "==============================================="
