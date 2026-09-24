# Cat-Dog Image Classification

A PyTorch computer-vision project for reproducible binary classification experiments on the Microsoft Cats vs. Dogs dataset.

## Experiments

All experiments used the same training and validation split, seed 42, batch size 32, Adam, and 10 training epochs. The scratch models and frozen ResNet18 used a learning rate of 0.001; full ResNet18 fine-tuning used 0.0001. The test split remained untouched during model development and validation.

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

![Validation accuracy versus trainable parameters](reports/figures/validation_accuracy_vs_parameters.png)

![Validation accuracy by model](reports/figures/validation_accuracy_by_model.png)

### Transfer Learning

Both transfer-learning experiments used ResNet18 with pretrained ImageNet weights and a new two-class classifier. Frozen-backbone training optimized only that classifier: the model had 11,177,538 total parameters, 1,026 trainable parameters, and reached 98.64% best validation accuracy. Full fine-tuning made all 11,177,538 parameters trainable, used a learning rate of 0.0001, and reached 98.93% best validation accuracy. Fine-tuning improved validation accuracy by approximately 0.29 percentage points in this experiment, while requiring more optimization time and producing a substantially larger Adam checkpoint.

Among the scratch models, replacing the original dense classifier with global average pooling greatly reduced parameter count but also reduced validation accuracy in these runs. Batch normalization slightly improved the compact GAP model, while classifier dropout reduced its validation result. Widening and deepening the convolutional backbone recovered performance. The first residual model retained high spatial resolution and had the longest runtime. Early downsampling made ResNetStyleCNN faster than that first residual model and produced the strongest scratch-model validation result. The pretrained ResNet18 experiments produced the strongest validation results overall. These measurements describe this specific split, seed, augmentation, and training configuration; they do not establish that transfer learning is universally superior. The test split has not been evaluated.
