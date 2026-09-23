# Cat-Dog Image Classification

A PyTorch computer-vision project for reproducible binary classification experiments on the Microsoft Cats vs. Dogs dataset.

## Experiments

All listed models were trained from scratch with the same training and validation split. Each run used seed 42, batch size 32, Adam with a learning rate of 0.001, and 10 training epochs. The test split remained untouched during model development and validation.

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

![Validation accuracy versus trainable parameters](reports/figures/validation_accuracy_vs_parameters.png)

![Validation accuracy by model](reports/figures/validation_accuracy_by_model.png)

Replacing the original dense classifier with global average pooling greatly reduced parameter count but also reduced validation accuracy in these runs. Batch normalization slightly improved the compact GAP model, while classifier dropout reduced its validation result. Widening and deepening the convolutional backbone recovered performance. The first residual model retained high spatial resolution and had the longest runtime. The ResNet-style model introduced early downsampling, ran substantially faster than that first residual model, and produced the strongest validation result among these experiments. These observations describe this dataset split and training configuration rather than universal model behavior.
