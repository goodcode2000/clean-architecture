#!/usr/bin/env python3
"""
CSV Data Provider Service
Serves CSV data from multiple prediction projects to display apps
Expandable for multiple first apps running on VPS
"""

import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from datetime import datetime
import glob

app = Flask(__name__)
CORS(app)

class CSVDataProvider:
    def __init__(self):
        # Configuration for multiple projects
        self.projects = {
            'btc-main72': {
                'name': 'BTC Main72 Project',
                'data_path': '/home/ubuntu/BTC/main72/clean-architecture/data',
                'files': {
                    'historical_real': 'historical_real.csv',
                    'predictions': 'predictions.csv'
                }
            }
            # Add more projects here as needed
            # 'eth-project': {
            #     'name': 'ETH Project',
            #     'data_path': '/home/ubuntu/ETH/project/data',
            #     'files': {
            #         'historical_real': 'historical_real.csv',
            #         'predictions': 'predictions.csv'
            #     }
            # }
        }
    
    def get_project_data_path(self, project_id):
        """Get data path for a specific project"""
        if project_id in self.projects:
            return self.projects[project_id]['data_path']
        return None
    
    def get_csv_file_path(self, project_id, file_type):
        """Get full path to CSV file"""
        if project_id not in self.projects:
            return None
        
        data_path = self.projects[project_id]['data_path']
        file_name = self.projects[project_id]['files'].get(file_type)
        
        if not file_name:
            return None
        
        return os.path.join(data_path, file_name)
    
    def read_csv_safely(self, file_path):
        """Safely read CSV file with error handling"""
        try:
            if not os.path.exists(file_path):
                return None, f"File not found: {file_path}"
            
            df = pd.read_csv(file_path)
            return df, None
        except Exception as e:
            return None, f"Error reading CSV: {str(e)}"
    
    def get_historical_data(self, project_id, limit=None):
        """Get historical real price data - latest 24 hours by default"""
        file_path = self.get_csv_file_path(project_id, 'historical_real')
        if not file_path:
            return None, "Project not found"
        
        df, error = self.read_csv_safely(file_path)
        if error:
            return None, error
        
        # Convert timestamp column to datetime for filtering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Get last 24 hours of data (288 points at 5-minute intervals)
        if limit is None:
            # Default: last 24 hours (288 points)
            df = df.tail(288)
        elif limit > 0:
            df = df.tail(limit)
        
        # Convert to API format
        data_points = []
        for _, row in df.iterrows():
            data_points.append({
                "timestamp": row['timestamp'].isoformat(),
                "price": float(row['price'])
            })
        
        return {
            "data": data_points,
            "total": len(data_points),
            "range": "24 hours" if limit is None else f"last {len(data_points)} points",
            "project": self.projects[project_id]['name'],
            "file_path": file_path,
            "last_updated": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }, None
    
    def get_prediction_data(self, project_id, limit=None):
        """Get prediction data - latest 24 hours by default"""
        file_path = self.get_csv_file_path(project_id, 'predictions')
        if not file_path:
            return None, "Project not found"
        
        df, error = self.read_csv_safely(file_path)
        if error:
            return None, error
        
        # Convert timestamp column to datetime for filtering
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate 24 hours ago from the latest prediction
        if len(df) > 0:
            latest_time = df['timestamp'].max()
            twenty_four_hours_ago = latest_time - pd.Timedelta(hours=24)
            
            # Filter for last 24 hours
            df_24h = df[df['timestamp'] >= twenty_four_hours_ago]
            
            # If less than 24 hours of data, use all available data
            if len(df_24h) == 0:
                df_24h = df  # Use all data if no data in last 24 hours
        else:
            df_24h = df  # Empty dataframe
        
        # Apply additional limit if specified
        if limit and len(df_24h) > limit:
            df_24h = df_24h.tail(limit)
        
        # Convert to API format
        predictions = []
        for _, row in df_24h.iterrows():
            pred_data = {
                "timestamp": row['timestamp'].isoformat(),
                "predicted_price": float(row['predicted_price'])
            }
            
            # Add optional fields if they exist
            if 'current_price' in row:
                pred_data['current_price'] = float(row['current_price'])
            if 'model' in row:
                pred_data['model'] = row['model']
            if 'confidence_lower' in row:
                pred_data['confidence_lower'] = float(row['confidence_lower'])
            if 'confidence_upper' in row:
                pred_data['confidence_upper'] = float(row['confidence_upper'])
            
            predictions.append(pred_data)
        
        hours_span = 24 if len(df_24h) > 0 else 0
        if len(df) > 0 and len(df_24h) > 0:
            actual_span = (df_24h['timestamp'].max() - df_24h['timestamp'].min()).total_seconds() / 3600
            hours_span = min(24, actual_span)
        
        return {
            "predictions": predictions,
            "total": len(predictions),
            "range": f"{hours_span:.1f} hours" if hours_span < 24 else "24 hours",
            "project": self.projects[project_id]['name'],
            "file_path": file_path,
            "last_updated": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
        }, None
    
    def get_latest_prediction(self, project_id):
        """Get the most recent prediction"""
        data, error = self.get_prediction_data(project_id, limit=1)
        if error:
            return None, error
        
        if not data['predictions']:
            return None, "No predictions available"
        
        return data['predictions'][0], None
    
    def get_project_status(self, project_id):
        """Get status of a specific project"""
        if project_id not in self.projects:
            return None, "Project not found"
        
        project = self.projects[project_id]
        status = {
            "project_id": project_id,
            "name": project['name'],
            "data_path": project['data_path'],
            "files": {}
        }
        
        # Check each file
        for file_type, file_name in project['files'].items():
            file_path = os.path.join(project['data_path'], file_name)
            if os.path.exists(file_path):
                stat = os.stat(file_path)
                status['files'][file_type] = {
                    "exists": True,
                    "size": stat.st_size,
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "path": file_path
                }
            else:
                status['files'][file_type] = {
                    "exists": False,
                    "path": file_path
                }
        
        return status, None

# Initialize provider
provider = CSVDataProvider()

# API Routes
@app.route('/api/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "CSV Data Provider",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "projects": list(provider.projects.keys())
    })

@app.route('/api/projects')
def list_projects():
    """List all available projects"""
    projects = []
    for project_id, project_info in provider.projects.items():
        projects.append({
            "id": project_id,
            "name": project_info['name'],
            "data_path": project_info['data_path']
        })
    
    return jsonify({
        "projects": projects,
        "total": len(projects)
    })

@app.route('/api/projects/<project_id>/status')
def project_status(project_id):
    """Get status of a specific project"""
    status, error = provider.get_project_status(project_id)
    if error:
        return jsonify({"error": error}), 404
    
    return jsonify(status)

@app.route('/api/projects/<project_id>/historical-data')
def historical_data(project_id):
    """Get historical real price data for a project"""
    limit = request.args.get('limit', type=int)
    
    data, error = provider.get_historical_data(project_id, limit)
    if error:
        return jsonify({"error": error}), 404
    
    return jsonify(data)

@app.route('/api/projects/<project_id>/predictions')
def prediction_data(project_id):
    """Get prediction data for a project"""
    limit = request.args.get('limit', type=int)
    
    data, error = provider.get_prediction_data(project_id, limit)
    if error:
        return jsonify({"error": error}), 404
    
    return jsonify(data)

@app.route('/api/projects/<project_id>/latest-prediction')
def latest_prediction(project_id):
    """Get latest prediction for a project"""
    data, error = provider.get_latest_prediction(project_id)
    if error:
        return jsonify({"error": error}), 404
    
    return jsonify(data)

@app.route('/api/projects/<project_id>/current-price')
def current_price(project_id):
    """Get current price from latest historical data"""
    data, error = provider.get_historical_data(project_id, limit=1)
    if error:
        return jsonify({"error": error}), 404
    
    if not data['data']:
        return jsonify({"error": "No price data available"}), 404
    
    latest_price = data['data'][0]
    return jsonify({
        "price": latest_price['price'],
        "timestamp": latest_price['timestamp'],
        "project": data['project']
    })

if __name__ == '__main__':
    print("🚀 Starting CSV Data Provider Service...")
    print("📊 Serving data from multiple prediction projects")
    print("🔗 Available projects:", list(provider.projects.keys()))
    print("="*60)
    
    # Start server on port 9000 to avoid conflicts
    app.run(host='0.0.0.0', port=9000, debug=False)