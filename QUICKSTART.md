# Quick Start Guide - App 1

## Installation (Ubuntu VPS)

```bash
# 1. Navigate to app directory
cd app1_ubuntu_vps

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create data directories
mkdir -p data models
```

## Running the Application

```bash
# Simple run
python main.py

# Or use the startup script
chmod +x run.sh
./run.sh
```

## What to Expect

1. **First Run**: 
   - Fetches 90 days of BTC historical data
   - Trains all ensemble models (takes 10-30 minutes depending on GPU)
   - Shows current real price

2. **Every 5 Minutes**:
   - Updates with latest data
   - Makes new prediction
   - Displays in terminal:
     - Current real price
     - Predicted price (5 min later)
     - Individual model predictions
     - Offset statistics

3. **Every 24 Hours**:
   - Automatically retrains models with latest data

## Terminal Output Example

```
======================================================================
Timestamp: 2024-01-15 14:30:00
----------------------------------------------------------------------
Current Real Price:                  $42,350.25
----------------------------------------------------------------------
Predicted Price (5 min):             $42,380.50
Expected Change:                        $30.25 (  0.07%)
----------------------------------------------------------------------
Individual Model Predictions:
  ETS              $42,365.00
  GARCH            $42,375.00
  LIGHTGBM         $42,385.00
  TCN              $42,390.00
  CNN              $42,370.00
----------------------------------------------------------------------
Offset Statistics (from 150 predictions):
  Mean Offset:   $35.20
  Median Offset: $32.10
  Std Offset:    $28.50
======================================================================
```

## Troubleshooting

**Issue: Models not training**
- Check internet connection for data fetching
- Ensure sufficient disk space
- Check Python version (3.8+)

**Issue: GPU not used**
- Install CUDA-compatible TensorFlow
- Check GPU availability: `nvidia-smi`

**Issue: Import errors**
- Run `pip install -r requirements.txt` again
- Check Python version compatibility

## Files Generated

- `data/predictions_history.csv`: All predictions for offset analysis
- `models/*.pkl`, `models/*.h5`, `models/*.lgb`: Saved model files

## Stopping the Application

Press `Ctrl+C` to stop gracefully.

