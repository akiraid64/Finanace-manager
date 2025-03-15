"""
Main entry point for the Finance Manager application
"""

import logging
import os
import sys
from pathlib import Path
import subprocess
import base64  # Fix missing import for app.py

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import pandas
        import matplotlib
        import sklearn
        import flask
        return True
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        return False

def try_build_cpp_module():
    """Attempt to build the C++ module if it doesn't exist"""
    # Check if the compiled module already exists
    module_path = Path(__file__).parent / 'src' / 'python' / 'finance.so'
    if module_path.exists():
        logger.info("C++ module already exists, skipping build")
        return True
    
    logger.info("C++ module not found, trying to build it")
    
    try:
        # Run the build script
        build_script = Path(__file__).parent / 'build_cpp.py'
        subprocess.run([sys.executable, str(build_script)], check=True)
        logger.info("C++ module built successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build C++ module: {e}")
        logger.warning("Will use Python fallback implementation")
        return False
    except Exception as e:
        logger.error(f"Unexpected error building C++ module: {e}")
        logger.warning("Will use Python fallback implementation")
        return False

def create_data_directory():
    """Create data directory if it doesn't exist"""
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    logger.info(f"Data directory: {data_dir}")

# Initialize application
logger.info("Starting Finance Manager application")

# Check dependencies
if not check_dependencies():
    logger.error("Missing required dependencies")
    sys.exit(1)

# Create data directory
create_data_directory()

# Try to build C++ module (will fall back to Python if it fails)
try_build_cpp_module()

# Import the Flask app
from app import app

# For running directly
if __name__ == "__main__":
    logger.info("Starting web server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
