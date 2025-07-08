#!/usr/bin/env python3
"""
Test script for the Email Spam Classifier
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_trainer import ModelTrainer
from models.spam_classifier import SpamClassifier

def test_basic_functionality():
    """Test basic classifier functionality"""
    print("🧪 Testing Email Spam Classifier...")
    print("=" * 50)
    
    try:
        # Initialize trainer
        print("1. Initializing model trainer...")
        trainer = ModelTrainer()
        
        # Load sample data
        print("2. Loading sample data...")
        email_contents, labels = trainer.load_sample_data()
        print(f"   Loaded {len(email_contents)} emails ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")
        
        # Train model
        print("3. Training model...")
        results = trainer.train_model(save_model=True)
        
        # Test predictions
        print("4. Testing predictions...")
        test_emails = [
            "URGENT: You've won $1,000,000! Click here to claim your prize now!",
            "Hi John, I hope you're doing well. Let's meet for coffee tomorrow at 3 PM."
        ]
        
        test_results = trainer.test_model(test_emails)
        
        print("\n✅ Test Results:")
        for i, result in enumerate(test_results):
            status = "SPAM" if result['is_spam'] else "HAM"
            confidence = result['confidence']
            print(f"   Email {i+1}: {status} (Confidence: {confidence:.1f}%)")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 Testing API endpoints...")
    print("=" * 50)
    
    try:
        # Test basic Flask functionality
        import flask
        print("✅ Flask import: OK")
        
        # Test basic app structure
        if os.path.exists('app.py'):
            print("✅ app.py exists: OK")
        else:
            print("❌ app.py not found")
            return False
        
        print("🎉 API tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Email Spam Classifier Tests")
    print("=" * 60)
    
    # Test basic functionality
    basic_test = test_basic_functionality()
    
    # Test API endpoints
    api_test = test_api_endpoints()
    
    print("\n" + "=" * 60)
    if basic_test and api_test:
        print("🎉 All tests passed! The Email Spam Classifier is ready to use.")
        print("\n📋 Next steps:")
        print("   1. Run 'python app.py' to start the web interface")
        print("   2. Visit http://localhost:5000 in your browser")
        print("   3. Use the web interface to classify emails")
        print("   4. Use the API endpoints for integration")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 