# GHI Forecasting

A machine learning project for Global Horizontal Irradiance (GHI) forecasting with Comet.ml integration for experiment tracking.

## Setup

1. Create and activate a conda environment:
```bash
# Create environment with Python 3.8.13
conda create -n ghifo_py38 python=3.8.13
conda activate ghifo_py38

# Install dependencies
pip install -r requirements.txt
```

2. Set up Comet.ml:
   - Create an account at [Comet.ml](https://www.comet.ml) if you don't have one
   - Get your API key from your Comet.ml account settings
   - Replace `YOUR-API-KEY` and `YOUR-WORKSPACE-NAME` in `train.py` with your actual values

3. Run the training script:
```bash
python train.py
```

## Environment Details

- Python version: 3.8.13
- TensorFlow version: 2.6.0
- Keras version: 2.6.0

## Features

- Experiment tracking with Comet.ml
- Automatic logging of:
  - Model parameters
  - Training and validation metrics
  - Feature importance
  - Dataset statistics

## Project Structure

- `requirements.txt`: Project dependencies
- `train.py`: Example training script with Comet.ml integration
