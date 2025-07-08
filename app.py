from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import os
import json
from datetime import datetime
import traceback

from models.spam_classifier import SpamClassifier
from models.model_trainer import ModelTrainer
from utils.text_preprocessor import TextPreprocessor
from utils.email_parser import EmailParser

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
CORS(app)

# Initialize components
classifier = None
model_trainer = ModelTrainer()
text_preprocessor = TextPreprocessor()
email_parser = EmailParser()

def load_classifier():
    """Load the trained classifier"""
    global classifier
    try:
        if classifier is None:
            classifier = SpamClassifier()
            if os.path.exists('models/saved/spam_classifier_models.pkl'):
                classifier.load_models('models/saved/spam_classifier')
                print("Loaded existing trained model")
            else:
                print("No trained model found. Please train the model first.")
        return classifier
    except Exception as e:
        print(f"Error loading classifier: {str(e)}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/classify', methods=['GET', 'POST'])
def classify_email():
    """Classify email content"""
    if request.method == 'POST':
        try:
            email_content = request.form.get('email_content', '').strip()
            
            if not email_content:
                flash('Please enter email content', 'error')
                return render_template('classify.html')
            
            # Load classifier
            clf = load_classifier()
            if clf is None or not clf.is_fitted:
                flash('Model not trained. Please train the model first.', 'error')
                return render_template('classify.html')
            
            # Make prediction
            result = clf.predict(email_content)
            
            # Prepare result for template
            classification_result = {
                'is_spam': result['is_spam'],
                'spam_probability': round(result['spam_probability'] * 100, 2),
                'ham_probability': round(result['ham_probability'] * 100, 2),
                'confidence': round(result['confidence'] * 100, 2),
                'individual_predictions': result['individual_predictions']
            }
            
            return render_template('classify.html', result=classification_result, email_content=email_content)
            
        except Exception as e:
            flash(f'Error during classification: {str(e)}', 'error')
            return render_template('classify.html')
    
    return render_template('classify.html')

@app.route('/train')
def train_page():
    """Training page"""
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the spam classifier"""
    try:
        # Train the model
        results = model_trainer.train_model(save_model=True)
        
        # Reload classifier
        global classifier
        classifier = None
        load_classifier()
        
        flash('Model trained successfully!', 'success')
        return jsonify({
            'success': True,
            'message': 'Model trained successfully',
            'results': results
        })
        
    except Exception as e:
        error_msg = f'Error training model: {str(e)}'
        flash(error_msg, 'error')
        return jsonify({
            'success': False,
            'message': error_msg
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for spam prediction"""
    try:
        data = request.get_json()
        
        if not data or 'email_content' not in data:
            return jsonify({
                'error': 'email_content is required'
            }), 400
        
        email_content = data['email_content'].strip()
        
        if not email_content:
            return jsonify({
                'error': 'email_content cannot be empty'
            }), 400
        
        # Load classifier
        clf = load_classifier()
        if clf is None or not clf.is_fitted:
            return jsonify({
                'error': 'Model not trained. Please train the model first.'
            }), 500
        
        # Make prediction
        result = clf.predict(email_content)
        
        # Format response
        response = {
            'is_spam': result['is_spam'],
            'spam_probability': round(result['spam_probability'], 4),
            'ham_probability': round(result['ham_probability'], 4),
            'confidence': round(result['confidence'], 4),
            'model_type': result['model_type'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/api/batch_predict', methods=['POST'])
def api_batch_predict():
    """API endpoint for batch spam prediction"""
    try:
        data = request.get_json()
        
        if not data or 'emails' not in data:
            return jsonify({
                'error': 'emails array is required'
            }), 400
        
        emails = data['emails']
        
        if not isinstance(emails, list) or len(emails) == 0:
            return jsonify({
                'error': 'emails must be a non-empty array'
            }), 400
        
        # Load classifier
        clf = load_classifier()
        if clf is None or not clf.is_fitted:
            return jsonify({
                'error': 'Model not trained. Please train the model first.'
            }), 500
        
        # Make batch predictions
        results = clf.predict_batch(emails)
        
        # Format response
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'email_index': i,
                'is_spam': result['is_spam'],
                'spam_probability': round(result['spam_probability'], 4),
                'ham_probability': round(result['ham_probability'], 4),
                'confidence': round(result['confidence'], 4)
            })
        
        response = {
            'results': formatted_results,
            'total_emails': len(emails),
            'spam_count': sum(1 for r in results if r['is_spam']),
            'ham_count': sum(1 for r in results if not r['is_spam']),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Batch prediction error: {str(e)}'
        }), 500

@app.route('/api/model_info')
def api_model_info():
    """API endpoint to get model information"""
    try:
        clf = load_classifier()
        
        if clf is None or not clf.is_fitted:
            return jsonify({
                'trained': False,
                'message': 'Model not trained'
            })
        
        # Get model summary
        summary = clf.get_model_summary()
        
        # Load training results if available
        training_results = {}
        if os.path.exists('models/saved/spam_classifier_results.json'):
            with open('models/saved/spam_classifier_results.json', 'r') as f:
                training_results = json.load(f)
        
        response = {
            'trained': True,
            'model_summary': summary,
            'training_results': training_results,
            'feature_count': len(clf.feature_extractor.get_feature_names()) if clf.feature_extractor.is_fitted else 0
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'error': f'Error getting model info: {str(e)}'
        }), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/dashboard')
def dashboard():
    """Dashboard page with model statistics"""
    try:
        clf = load_classifier()
        
        if clf is None or not clf.is_fitted:
            return render_template('dashboard.html', model_trained=False)
        
        # Get model information
        summary = clf.get_model_summary()
        
        # Load training results
        training_results = {}
        if os.path.exists('models/saved/spam_classifier_results.json'):
            with open('models/saved/spam_classifier_results.json', 'r') as f:
                training_results = json.load(f)
        
        return render_template('dashboard.html', 
                             model_trained=True,
                             summary=summary,
                             training_results=training_results)
        
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'error')
        return render_template('dashboard.html', model_trained=False)

@app.route('/upload', methods=['GET', 'POST'])
def upload_email():
    """Upload email file for classification"""
    if request.method == 'POST':
        try:
            if 'email_file' not in request.files:
                flash('No file uploaded', 'error')
                return render_template('upload.html')
            
            file = request.files['email_file']
            
            if file.filename == '':
                flash('No file selected', 'error')
                return render_template('upload.html')
            
            if file:
                # Read file content
                email_content = file.read().decode('utf-8', errors='ignore')
                
                # Parse email if it's in email format
                try:
                    email_data = email_parser.parse_email_string(email_content)
                    if 'error' not in email_data:
                        # Extract content for classification
                        email_content = email_parser.get_email_content_for_classification(email_data)
                except:
                    # Use raw content if parsing fails
                    pass
                
                # Load classifier
                clf = load_classifier()
                if clf is None or not clf.is_fitted:
                    flash('Model not trained. Please train the model first.', 'error')
                    return render_template('upload.html')
                
                # Make prediction
                result = clf.predict(email_content)
                
                # Prepare result
                classification_result = {
                    'is_spam': result['is_spam'],
                    'spam_probability': round(result['spam_probability'] * 100, 2),
                    'ham_probability': round(result['ham_probability'] * 100, 2),
                    'confidence': round(result['confidence'] * 100, 2),
                    'individual_predictions': result['individual_predictions']
                }
                
                return render_template('upload.html', result=classification_result, email_content=email_content[:500])
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return render_template('upload.html')
    
    return render_template('upload.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # Load classifier on startup
    load_classifier()
    
    print("Starting Email Spam Classifier...")
    print("Visit http://localhost:5000 to use the web interface")
    print("API endpoints available at:")
    print("  - POST /api/predict")
    print("  - POST /api/batch_predict")
    print("  - GET /api/model_info")
    print("  - GET /api/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 