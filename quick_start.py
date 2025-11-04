#!/usr/bin/env python3
"""
Quick Start Script for BTC Predictor
Automatically sets up and starts the system
"""
import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Need Python 3.8+")
        return False

def setup_environment():
    """Set up virtual environment and install dependencies"""
    print("\n🏗️  Setting up environment...")
    
    # Create virtual environment if it doesn't exist
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Determine activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install minimal requirements first
    install_cmd = f"{pip_cmd} install -r requirements_minimal.txt"
    if not run_command(install_cmd, "Installing minimal requirements"):
        print("⚠️  Trying with basic packages...")
        basic_packages = "pandas numpy requests flask flask-cors scikit-learn python-dotenv loguru"
        if not run_command(f"{pip_cmd} install {basic_packages}", "Installing basic packages"):
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["data", "logs", "models", "models/saved", "models/ensemble", "models/individual"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    return True

def run_tests():
    """Run system tests"""
    print("\n🧪 Running system tests...")
    
    # Determine python command
    python_cmd = "venv\\Scripts\\python" if os.name == 'nt' else "venv/bin/python"
    
    return run_command(f"{python_cmd} test_complete_system.py", "System tests")

def start_system(mode="simple"):
    """Start the BTC predictor system"""
    print(f"\n🚀 Starting BTC Predictor ({mode} mode)...")
    
    # Determine python command
    python_cmd = "venv\\Scripts\\python" if os.name == 'nt' else "venv/bin/python"
    
    if mode == "simple":
        script = "simple_predictor.py"
    else:
        script = "main.py"
    
    print(f"Starting {script}...")
    print("Press Ctrl+C to stop the system")
    print("="*60)
    
    try:
        # Run the system
        subprocess.run(f"{python_cmd} {script}", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 System stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ System failed to start: {e}")

def main():
    """Main setup and start function"""
    print("="*60)
    print("🚀 BTC PREDICTOR QUICK START")
    print("="*60)
    
    # Check Python version
    if not check_python():
        return
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but continuing...")
    
    # Ask user which mode to start
    print("\n" + "="*60)
    print("🎯 READY TO START!")
    print("="*60)
    print("Choose startup mode:")
    print("1. Simple mode (recommended for first run)")
    print("2. Full system mode (all ML models)")
    print("3. Just run tests and exit")
    
    while True:
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == "1":
            start_system("simple")
            break
        elif choice == "2":
            start_system("full")
            break
        elif choice == "3":
            print("✅ Setup complete. You can manually start with:")
            print("   Simple: python simple_predictor.py")
            print("   Full:   python main.py")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()