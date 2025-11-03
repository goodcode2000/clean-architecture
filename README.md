# Prediction Engine Deployment

This folder contains the BTC price prediction engine service. Follow these steps to deploy it on a VPS:

1. Clone only this folder to your VPS:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/goodcode2000/clean-architecture.git
cd clean-architecture
git sparse-checkout set prediction-engine
```

2. Navigate to the prediction-engine folder:
```bash
cd prediction-engine
```

3. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

4. Activate the virtual environment:
```bash
source venv/bin/activate
```

5. Start the prediction engine:
```bash
python main.py --mode api --host 0.0.0.0 --port 8000
```

## Requirements

- Python 3.8 or higher
- Virtual environment support (python3-venv)
- Git

## Service Configuration (Optional)

To run as a systemd service, create `/etc/systemd/system/prediction-engine.service`:

```ini
[Unit]
Description=BTC Prediction Engine
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/prediction-engine
Environment=PATH=/path/to/prediction-engine/venv/bin:$PATH
ExecStart=/path/to/prediction-engine/venv/bin/python main.py --mode api --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Then enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable prediction-engine
sudo systemctl start prediction-engine
```

## Monitoring

Check the service status:
```bash
sudo systemctl status prediction-engine
```

View logs:
```bash
tail -f logs/prediction_engine.log
```