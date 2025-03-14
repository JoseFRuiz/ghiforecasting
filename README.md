# GHI Forecasting

A machine learning project for Global Horizontal Irradiance (GHI) forecasting with Comet.ml integration for experiment tracking.

## Setup

### Option 1: Using Conda (Recommended)

1. Create and activate the Conda environment:
```bash
conda env create -f environment.yml
conda activate ghi-forecast
```

### Option 2: Using pip

1. Install the required dependencies:
```bash
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

## Features

- Experiment tracking with Comet.ml
- Automatic logging of:
  - Model parameters
  - Training and validation metrics
  - Feature importance
  - Dataset statistics

## Project Structure

- `environment.yml`: Conda environment specification
- `requirements.txt`: Project dependencies (for pip)
- `train.py`: Example training script with Comet.ml integration
