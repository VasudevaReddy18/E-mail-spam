import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from models.feature_extractor import FeatureExtractor
from utils.text_preprocessor import TextPreprocessor
from utils.email_parser import EmailParser

class SpamClassifier:
    """
    Advanced spam classifier using ensemble of multiple algorithms
    """
    
    def __init__(self, model_type='ensemble'):
        """Initialize the spam classifier"""
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.email_parser = EmailParser()
        
        # Initialize models
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        
        # Model configurations
        self.model_configs = {
            'naive_bayes': {
                'model': MultinomialNB(alpha=1.0),
                'weight': 0.15
            },
            'svm': {
                'model': SVC(kernel='rbf', probability=True, random_state=42),
                'weight': 0.25
            },
            'random_forest': {
                'model': RandomForestClassifier(n_estimators=100, random_state=42),
                'weight': 0.20
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'weight': 0.15
            },
            'neural_network': {
                'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500),
                'weight': 0.25
            }
        }
        
        # Training data
        self.X_full_train = None
        self.X_full_test = None
        self.X_nonneg_train = None
        self.X_nonneg_test = None
        self.y_train = None
        self.y_test = None
        
        # Performance metrics
        self.performance_metrics = {}
        
        # Fitted flag
        self.is_fitted = False
    
    def prepare_data(self, email_contents: List[str], labels: List[int], 
                    email_data_list: Optional[List[Dict]] = None, test_size: float = 0.2):
        """
        Prepare training and testing data
        """
        # Split data first
        y = np.array(labels)
        email_contents = np.array(email_contents)
        if email_data_list is not None:
            email_data_list = np.array(email_data_list)
        X_train_emails, X_test_emails, y_train, y_test = train_test_split(
            email_contents, y, test_size=test_size, random_state=42, stratify=y
        )
        if email_data_list is not None:
            X_train_data, X_test_data = train_test_split(
                email_data_list, test_size=test_size, random_state=42, stratify=y
            )
        else:
            X_train_data = X_test_data = None

        # Fit feature extractor only on training set
        self.feature_extractor.fit(list(X_train_emails), list(X_train_data) if X_train_data is not None else None)
        # Transform both train and test sets
        X_full_train = self.feature_extractor.transform(list(X_train_emails), list(X_train_data) if X_train_data is not None else None)
        X_full_test = self.feature_extractor.transform(list(X_test_emails), list(X_test_data) if X_test_data is not None else None)
        X_nonneg_train = self.feature_extractor.get_non_negative_feature_matrix(list(X_train_emails))
        X_nonneg_test = self.feature_extractor.get_non_negative_feature_matrix(list(X_test_emails))

        # Scale full features
        X_full_train_scaled = self.scaler.fit_transform(X_full_train)
        X_full_test_scaled = self.scaler.transform(X_full_test)
        print(f"[DEBUG] StandardScaler fitted. Mean (first 5): {self.scaler.mean_[:5]}")

        # Store all splits
        self.X_full_train = X_full_train_scaled
        self.X_full_test = X_full_test_scaled
        self.X_nonneg_train = X_nonneg_train
        self.X_nonneg_test = X_nonneg_test
        self.y_train = y_train
        self.y_test = y_test

        return self.X_full_train, self.X_full_test, self.X_nonneg_train, self.X_nonneg_test, self.y_train, self.y_test
    
    def _check_scaler_fitted(self):
        if not hasattr(self.scaler, 'mean_'):
            import traceback
            print('[ERROR] StandardScaler instance is not fitted!')
            raise RuntimeError('StandardScaler instance is not fitted yet. Call fit before using this estimator.\n' + ''.join(traceback.format_stack()))

    def train_single_model(self, model_name: str) -> Dict:
        """
        Train a single model and return performance metrics
        """
        model_config = self.model_configs[model_name]
        model = model_config['model']
        
        # Select features
        if model_name == 'naive_bayes':
            X_train = self.X_nonneg_train
            X_test = self.X_nonneg_test
            print(f"[DEBUG] MultinomialNB X_train min: {X_train.min()}, max: {X_train.max()}")
        else:
            self._check_scaler_fitted()
            X_train = self.X_full_train
            X_test = self.X_full_test
        
        # Train the model
        model.fit(X_train, self.y_train)
        
        # Store the model
        self.models[model_name] = model
        self.model_weights[model_name] = model_config['weight']
        
        # Evaluate performance
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='binary')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        self.performance_metrics[model_name] = metrics
        
        return metrics
    
    def train_all_models(self, email_contents: List[str], labels: List[int], 
                        email_data_list: Optional[List[Dict]] = None):
        """
        Train all models and create ensemble
        """
        # Prepare data (scaler is fitted and features are scaled here)
        self.prepare_data(email_contents, labels, email_data_list)
        # Do NOT fit or transform scaler again here! All features are already scaled.
        
        # Train each model
        for model_name in self.model_configs.keys():
            print(f"[DEBUG] About to train {model_name}...")
            metrics = self.train_single_model(model_name)
            print(f"[DEBUG] Finished training {model_name}. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        self.is_fitted = True
        
        return self.performance_metrics
    
    def predict(self, email_content: str, email_data: Optional[Dict] = None) -> Dict:
        """
        Predict if email is spam
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        # Extract features
        features_full = self.feature_extractor.extract_features(email_content, email_data).reshape(1, -1)
        features_nonneg = self.feature_extractor.get_non_negative_features(email_content).reshape(1, -1)
        # Prepare a dict of features for each model
        self._check_scaler_fitted()
        features_dict = {
            'naive_bayes': features_nonneg,
            'svm': self.scaler.transform(features_full),
            'random_forest': self.scaler.transform(features_full),
            'logistic_regression': self.scaler.transform(features_full),
            'neural_network': self.scaler.transform(features_full)
        }
        # Make ensemble prediction
        predictions = []
        probabilities = []
        individual_predictions = {}
        for model_name, model in self.models.items():
            feats = features_dict[model_name]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feats)[0, 1]
            else:
                proba = model.predict(feats)[0]
            pred = int(model.predict(feats)[0])
            predictions.append(pred * self.model_weights[model_name])
            probabilities.append(proba * self.model_weights[model_name])
            individual_predictions[model_name] = {
                'prediction': pred,
                'probability': float(proba)
            }
        # Weighted average
        ensemble_pred = np.sum(predictions)
        ensemble_proba = np.sum(probabilities)
        final_prediction = int(ensemble_pred > 0.5)
        result = {
            'is_spam': bool(final_prediction),
            'spam_probability': float(ensemble_proba),
            'ham_probability': float(1 - ensemble_proba),
            'confidence': float(abs(ensemble_proba - 0.5) * 2),
            'individual_predictions': individual_predictions,
            'model_type': self.model_type
        }
        return result
    
    def predict_batch(self, email_contents: List[str], 
                     email_data_list: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Predict spam for multiple emails
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        # Extract features for all emails
        features_full = self.feature_extractor.transform(email_contents, email_data_list)
        features_nonneg = self.feature_extractor.get_non_negative_feature_matrix(email_contents)
        # Prepare a dict of features for each model
        self._check_scaler_fitted()
        print(f"[DEBUG] Using StandardScaler. Fitted: {'mean_' in dir(self.scaler)}")
        features_dict = {
            'naive_bayes': features_nonneg,
            'svm': self.scaler.transform(features_full),
            'random_forest': self.scaler.transform(features_full),
            'logistic_regression': self.scaler.transform(features_full),
            'neural_network': self.scaler.transform(features_full)
        }
        # Make ensemble predictions
        results = []
        for i in range(len(email_contents)):
            predictions = []
            probabilities = []
            for model_name, model in self.models.items():
                feats = features_dict[model_name][i].reshape(1, -1)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(feats)[0, 1]
                else:
                    proba = model.predict(feats)[0]
                pred = int(model.predict(feats)[0])
                predictions.append(pred * self.model_weights[model_name])
                probabilities.append(proba * self.model_weights[model_name])
            ensemble_pred = np.sum(predictions)
            ensemble_proba = np.sum(probabilities)
            final_prediction = int(ensemble_pred > 0.5)
            result = {
                'is_spam': bool(final_prediction),
                'spam_probability': float(ensemble_proba),
                'ham_probability': float(1 - ensemble_proba),
                'confidence': float(abs(ensemble_proba - 0.5) * 2),
                'model_type': self.model_type
            }
            results.append(result)
        return results
    
    def ensemble_predict(self, X_full: np.ndarray, X_nonneg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble prediction using weighted voting
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before making predictions")
        self._check_scaler_fitted()
        print(f"[DEBUG] Using StandardScaler in ensemble_predict. Fitted: {'mean_' in dir(self.scaler)}")
        predictions = []
        probabilities = []
        for model_name, model in self.models.items():
            weight = self.model_weights[model_name]
            if model_name == 'naive_bayes':
                feats = X_nonneg
            else:
                feats = X_full
            pred = model.predict(feats)
            predictions.append(pred * weight)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feats)[:, 1] * weight
            else:
                proba = pred * weight
            probabilities.append(proba)
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        ensemble_proba = np.sum(probabilities, axis=0)
        # Convert to binary predictions
        final_predictions = (ensemble_pred > 0.5).astype(int)
        return final_predictions, ensemble_proba

    def evaluate_performance(self) -> Dict:
        """
        Evaluate overall ensemble performance
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before evaluation")
        self._check_scaler_fitted()
        print(f"[DEBUG] Using StandardScaler in evaluate_performance. Fitted: {'mean_' in dir(self.scaler)}")
        # Make ensemble predictions on test set
        predictions, probabilities = self.ensemble_predict(self.X_full_test, self.X_nonneg_test)
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, predictions, average='binary')
        # Confusion matrix
        cm = confusion_matrix(self.y_test, predictions)
        # Store ensemble metrics
        ensemble_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }
        self.performance_metrics['ensemble'] = ensemble_metrics
        return ensemble_metrics
    
    def get_feature_importance(self, model_name: str = 'random_forest') -> Dict:
        """
        Get feature importance from Random Forest model
        """
        if not self.is_fitted or model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            feature_names = self.feature_extractor.get_feature_names()
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return {}
    
    def save_models(self, filepath: str):
        """
        Save all trained models and feature extractor
        """
        if not self.is_fitted:
            raise ValueError("Models must be trained before saving")
        
        # Save feature extractor
        self.feature_extractor.save(filepath + '_feature_extractor.pkl')
        
        # Save models and scaler
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'performance_metrics': self.performance_metrics,
            'model_type': self.model_type
        }
        
        with open(filepath + '_models.pkl', 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_models(self, filepath: str):
        """
        Load trained models and feature extractor
        """
        # Load feature extractor
        self.feature_extractor = FeatureExtractor.load(filepath + '_feature_extractor.pkl')
        
        # Load models
        with open(filepath + '_models.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.model_weights = model_data['model_weights']
        self.scaler = model_data['scaler']
        self.performance_metrics = model_data['performance_metrics']
        self.model_type = model_data['model_type']
        
        self.is_fitted = True
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all models and their performance
        """
        if not self.is_fitted:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'number_of_models': len(self.models),
            'models': list(self.models.keys()),
            'performance': {}
        }
        
        for model_name, metrics in self.performance_metrics.items():
            summary['performance'][model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0)
            }
        
        return summary 