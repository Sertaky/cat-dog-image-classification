# Cat-Dog Image Classification

An end-to-end PyTorch computer vision case study for binary classification on the Microsoft/Kaggle Cats vs Dogs dataset. The repository covers dataset auditing, reproducible splitting, a custom `Dataset` and `DataLoader` pipeline, augmentation and normalization, controlled CNN experiments with global average pooling (GAP), BatchNorm, Dropout, and residual connections, transfer learning with ResNet18, final held-out test evaluation, and GPU inference benchmarking.

## Dataset and Reproducible Splits

The local dataset follows the original class-directory structure:

```text
PetImages/
├── Cat/
└── Dog/
```

The audited dataset contains 24,998 images, balanced between 12,499 cats and 12,499 dogs. A deterministic, stratified split generated with seed 42 is stored in CSV manifests:

| Split | Cat | Dog | Total |
|---|---:|---:|---:|
| Train | 8,749 | 8,749 | 17,498 |
| Validation | 1,875 | 1,875 | 3,750 |
| Test | 1,875 | 1,875 | 3,750 |

Raw images remain immutable. The manifests in `splits/` define membership without copying, moving, repairing, or rewriting source files. A full loading diagnostic found one non-fatal Pillow warning for `PetImages/Dog/9041.jpg`: `UserWarning: Truncated File Read`. The image remains in the training split, and the raw file was not modified or automatically excluded.

## Model Development

The experiment sequence was designed to isolate architectural changes:

`SimpleCNN` → GAP → BatchNorm → Dropout → wider CNN → deeper CNN → `ResidualCNN` → ResNet-style CNN → frozen pretrained ResNet18 → fine-tuned ResNet18

Every row below comes from `reports/experiment_summary.csv`. All experiments used the same training and validation manifests, seed 42, batch size 32, Adam, and 10 epochs. Scratch models and frozen ResNet18 used a learning rate of 0.001; full ResNet18 fine-tuning used 0.0001. The test split was not used during model development or checkpoint selection.

| Model | Parameters | Best Epoch | Best Validation Accuracy | Best Validation Loss | Runtime (minutes) |
|---|---:|---:|---:|---:|---:|
| SimpleCNN | 6,446,498 | 10 | 82.77% | 0.369600 | 24.95 |
| SimpleCNNGAP | 32,162 | 8 | 76.72% | 0.494508 | 14.87 |
| SimpleCNNGAPBatchNorm | 32,386 | 9 | 77.39% | 0.486278 | 14.96 |
| SimpleCNNGAPBatchNormDropout | 32,386 | 9 | 74.56% | 0.527677 | 13.38 |
| WideCNNGAPBatchNorm | 110,466 | 10 | 78.85% | 0.451519 | 18.44 |
| DeepCNNGAPBatchNorm | 422,530 | 8 | 81.49% | 0.410582 | 21.63 |
| ResidualCNN | 1,226,914 | 10 | 76.29% | 0.494965 | 52.59 |
| ResNetStyleCNN | 4,906,818 | 10 | 88.77% | 0.267854 | 20.33 |
| ResNet18Frozen | 11,177,538 | 10 | 98.64% | 0.041052 | 16.46 |
| ResNet18FineTune | 11,177,538 | 7 | 98.93% | 0.031699 | 22.84 |

![Validation accuracy versus model parameters](reports/figures/validation_accuracy_vs_parameters.png)

![Validation accuracy by model](reports/figures/validation_accuracy_by_model.png)

In this experiment series, the original dense classifier dominated the SimpleCNN parameter count. GAP reduced parameters substantially but initially reduced validation accuracy. BatchNorm produced a small improvement for the compact GAP model, while Dropout reduced validation performance in that specific configuration. Widening and deepening the convolutional backbone recovered accuracy. The first residual model retained high spatial resolution and was computationally expensive; early downsampling made the ResNet-style scratch model more effective. ImageNet transfer learning produced the strongest validation results. These are measurements from this dataset, split, seed, augmentation, and training configuration, not universal architecture rules.

## Final Selected Model

The selected model is `ResNet18Transfer`, fully fine-tuned from ImageNet-pretrained ResNet18 weights. Selection used best validation accuracy only.

| Configuration | Value |
|---|---:|
| Best checkpoint epoch | 7 |
| Validation accuracy | 98.9333% |
| Validation loss | 0.0316986 |
| Learning rate | 0.0001 |
| Total/trainable parameters | 11,177,538 |

All 11,177,538 parameters were trainable during fine-tuning.

## Final Test Evaluation

The held-out test split was evaluated once after model selection was complete. No subsequent tuning used the test results.

| Metric | Result |
|---|---:|
| Test samples | 3,750 |
| Loss | 0.037962 |
| Accuracy | 98.5867% |
| Correct predictions | 3,697 / 3,750 |

### Confusion Matrix

| Actual class | Predicted Cat | Predicted Dog |
|---|---:|---:|
| Cat | 1,853 | 22 |
| Dog | 31 | 1,844 |

### Per-Class Metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Cat | 0.983546 | 0.988267 | 0.985901 |
| Dog | 0.988210 | 0.983467 | 0.985833 |
| Macro average | 0.985878 | 0.985867 | 0.985867 |

## Inference Benchmark

The benchmark compares three checkpoints on an NVIDIA GeForce RTX 3050 4GB Laptop GPU with PyTorch 2.11.0+cu128, CUDA 12.8, `float32`, and 224 × 224 inputs. Timings cover model forward passes only; preprocessing, data loading, and host-to-device transfers are excluded.

### Batch Size 1

| Model | Parameters | Checkpoint (bytes) | Latency/image (ms) | Throughput (images/s) | Peak GPU allocation (bytes) |
|---|---:|---:|---:|---:|---:|
| DeepCNNGAPBatchNorm | 422,530 | 5,103,625 | 1.8539 | 539.39 | 24,715,776 |
| ResNetStyleCNN | 4,906,818 | 58,965,091 | 2.3655 | 422.74 | 40,002,048 |
| ResNet18Transfer | 11,177,538 | 134,268,861 | 4.0259 | 248.39 | 65,106,432 |

### Batch Size 32

| Model | Latency/image (ms) | Throughput (images/s) | Peak GPU allocation (bytes) |
|---|---:|---:|---:|
| DeepCNNGAPBatchNorm | 1.4892 | 671.51 | 441,577,984 |
| ResNetStyleCNN | 0.9650 | 1,036.22 | 254,017,536 |
| ResNet18Transfer | 1.5967 | 626.30 | 279,121,920 |

DeepCNNGAPBatchNorm had the lowest single-image latency. ResNetStyleCNN had the highest batch-32 throughput. Fine-tuned ResNet18 delivered the strongest accuracy, while also having the largest checkpoint and highest single-image latency among these three models. Deployment selection therefore depends on the target latency, throughput, memory, and accuracy constraints rather than one overall winner.

## Reproducibility

- Python: 3.12.14
- Dependencies: `requirements.txt`
- Experiment seed: 42
- Dataset membership: committed CSV manifests in `splits/`
- Recorded experiment results: CSV and JSON files in `reports/`

Run commands from the repository root. Module execution keeps project imports available and preserves the Windows multiprocessing entry-point guard used by the training scripts.

```powershell
python -m scripts.train_resnet18_finetune
python -m scripts.evaluate_resnet18_test
python -m scripts.benchmark_inference
python scripts/plot_experiment_results.py
```

The portable requirements file does not encode a platform-specific CUDA wheel suffix. For a CUDA-enabled PyTorch build, use the official PyTorch installation selector for the target operating system and CUDA configuration, then install the remaining requirements.

## Repository Structure

```text
cat-dog-image-classification/
├── src/
│   ├── datasets/
│   ├── transforms/
│   ├── dataloaders/
│   ├── models/
│   └── training/
├── scripts/
├── splits/
├── reports/
│   └── figures/
├── checkpoints/       # ignored model artifacts
├── PetImages/         # ignored immutable raw data
├── requirements.txt
└── README.md
```

## Key Takeaways

- Architecture mattered more than parameter count alone: models with similar or larger parameter counts produced materially different accuracy and runtime.
- Parameter count and runtime were not interchangeable; spatial resolution and execution pattern also affected cost.
- Transfer learning was highly effective on this dataset. Full fine-tuning improved validation accuracy over frozen features by about 0.29 percentage points in these runs.
- Validation selected the checkpoint, while the test split remained held out until the final evaluation.
- Deployment decisions require measured latency, throughput, memory, checkpoint size, and accuracy trade-offs rather than accuracy alone.
