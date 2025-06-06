# DCST-Net: A Decoupled Collaborative Spatio-Temporal Framework for Airfare Price Forecasting
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.10+](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)](https://pytorch.org/)

This repository implements an advanced airfare prediction system using hybrid graph neural networks and temporal decomposition techniques.

## Key Features

- 🛠 **Multi-View Graph Modeling**: Hierarchical GNN for station/route/flight relationships
- ⏳ **Dual Temporal Decoupling**: Separate trend/periodic components using diffusion models and pyramid LSTM
- 🤖 **Adaptive Fusion**: Cross-attention mechanism with gated fusion
- 📈 **MAPE Optimization**: Custom loss function for business-oriented evaluation
- 🚀 **Production Ready**: Supports distributed training and mixed precision

## Requirements

### Hardware
- NVIDIA GPU (RTX 2080 Ti or higher recommended)
- 16GB+ VRAM for full model training

### Software
```bash
Python 3.8+
PyTorch 1.12.1+cu113
torch-geometric 2.2.0
torchvision 0.13.1
numpy 1.23.5
pandas 1.5.2
scikit-learn 1.1.3
```

## Dataset Preparation
### Data Structure
```bash
data/
├── graph/
│   ├── stations.csv         # Station metadata
│   ├── routes.csv           # Route connections  
│   └── flights.csv          # Flight information
└── temporal/
    ├── series_a/            # Fixed departure series
    ├── series_b/            # Fixed purchase interval series
    └── prices.csv           # Historical price records
```
### Preprocessing
```bash
python prepare_data.py \
  --graph_dir data/graph \
  --temporal_dir data/temporal \
  --output_dir processed_data
```

## Training
### Basic Training
```bash
python train.py \
  --gnn_dim 128 \
  --pred_steps 7 \
  --batch_size 32 \
  --lr 1e-4 \
  --max_epochs 100
```

### Advanced Options
```bash
python train.py \
  --use_amp \                # Enable mixed precision
  --num_gpus 2 \             # Multi-GPU training
  --window_size 30 \         # Historical lookback window
  --pyramid_levels 7 14 21 \ # Periodic analysis scales
  --diffusion_steps 1000     # Noise scheduling steps
```

## Prediction
### Generate Predictions
```bash
python predict.py \
  --model_checkpoint best_model.pt \
  --input_data test_samples.json \
  --output predictions.csv
```
