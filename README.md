# CSV Data Provider Service

A centralized service that reads CSV files from multiple prediction projects and serves the data via REST API to display apps.

## Features

- **Multi-project support**: Serves data from multiple first apps
- **CSV file reading**: Reads historical_real.csv and predictions.csv
- **REST API**: Clean endpoints for data access
- **Expandable**: Easy to add new projects
- **Error handling**: Robust file reading with error responses

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python app.py
```

## API Endpoints

### Health Check
```
GET /api/health
```

### List Projects
```
GET /api/projects
```

### Project Status
```
GET /api/projects/{project_id}/status
```

### Historical Data
```
GET /api/projects/{project_id}/historical-data?limit=1000
```

### Predictions
```
GET /api/projects/{project_id}/predictions?limit=50
```

### Latest Prediction
```
GET /api/projects/{project_id}/latest-prediction
```

### Current Price
```
GET /api/projects/{project_id}/current-price
```

## Configuration

Edit the `projects` dictionary in `app.py` to add new projects:

```python
self.projects = {
    'btc-main72': {
        'name': 'BTC Main72 Project',
        'data_path': '/home/ubuntu/BTC/main72/clean-architecture/data',
        'files': {
            'historical_real': 'historical_real.csv',
            'predictions': 'predictions.csv'
        }
    },
    'eth-project': {
        'name': 'ETH Project', 
        'data_path': '/home/ubuntu/ETH/project/data',
        'files': {
            'historical_real': 'historical_real.csv',
            'predictions': 'predictions.csv'
        }
    }
}
```

## Usage

1. **Deploy to VPS**: Upload this folder to your VPS
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Configure projects**: Edit project paths in `app.py`
4. **Run service**: `python app.py` (runs on port 9000)
5. **Update display apps**: Point them to this service instead of direct first app APIs

## Example Usage

```bash
# Check service health
curl http://localhost:9000/api/health

# Get BTC project data
curl http://localhost:9000/api/projects/btc-main72/historical-data

# Get latest prediction
curl http://localhost:9000/api/projects/btc-main72/latest-prediction
```