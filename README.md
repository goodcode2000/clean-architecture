# BTC Price Prediction System

## Overview
AI ensemble model for predicting Bitcoin prices 5 minutes ahead using multiple ML algorithms (ETS, SVR, Random Forest, LightGBM, LSTM).

## Project Structure
- `models/` - ML model implementations
- `data/` - Data collection and storage
- `services/` - Core business logic services
- `api/` - REST API endpoints
- `config/` - Configuration files
- `tests/` - Unit and integration tests
- `logs/` - Application logs

## Setup
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate` (Linux) or `venv\Scripts\activate` (Windows)
3. Install dependencies: `pip install -r requirements.txt`

## Usage
Run the prediction system: `python main.py`