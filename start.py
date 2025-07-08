#!/usr/bin/env python3
"""
Startup script for Email Spam Classifier
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask', 'numpy', 'pandas', 'scikit-learn', 'nltk', 
        'beautifulsoup4', 'joblib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    try:
        import nltk
        
        # Download required NLTK data
        nltk_data = ['punkt', 'stopwords', 'wordnet']
        for data in nltk_data:
            try:
                nltk.data.find(f'tokenizers/{data}')
            except LookupError:
                print(f"   Downloading {data}...")
                nltk.download(data, quiet=True)
        
        print("✅ NLTK data downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"⚠️  Warning: Could not download NLTK data: {e}")
        print("   The application may still work, but with limited functionality.")
        return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        'models/saved',
        'data',
        'static',
        'templates'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directories created successfully!")

def start_application():
    """Start the Flask application"""
    print("🚀 Starting Email Spam Classifier...")
    print("=" * 60)
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ app.py not found. Please ensure you're in the correct directory.")
        return False
    
    # Start the application
    try:
        print("🌐 Starting web server...")
        print("📱 The application will be available at: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 60)
        
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎯 Email Spam Classifier - Startup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Download NLTK data
    download_nltk_data()
    
    # Create directories
    create_directories()
    
    # Start application
    start_application()

if __name__ == "__main__":
    main() 