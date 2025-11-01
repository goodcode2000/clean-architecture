"""Setup script for BTC Predictor application."""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    try:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install dependencies: {e}")
        return False

def create_virtual_env():
    """Create virtual environment."""
    try:
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", "venv"])
        print("✓ Virtual environment created")
        print("To activate: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Linux)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create virtual environment: {e}")
        return False

def main():
    """Main setup function."""
    print("BTC Predictor Setup")
    print("==================")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("✗ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    from config.config import Config
    Config.create_directories()
    print("✓ Project directories created")
    
    print("\nSetup completed! Next steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: venv\\Scripts\\activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Run application: python main.py")

if __name__ == "__main__":
    main()