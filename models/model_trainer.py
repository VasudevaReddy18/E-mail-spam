import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import traceback
import argparse

from models.spam_classifier import SpamClassifier
from utils.text_preprocessor import TextPreprocessor
from utils.email_parser import EmailParser
from models.feature_extractor import FeatureExtractor

class ModelTrainer:
    """
    Model trainer for spam classification
    """
    
    def __init__(self, data_path: str = 'data/spam_data.csv'):
        """Initialize the model trainer"""
        self.data_path = data_path
        self.classifier = SpamClassifier()
        self.text_preprocessor = TextPreprocessor()
        self.email_parser = EmailParser()
        
        # Training results
        self.training_results = {}
        self.best_model = None
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('models/saved', exist_ok=True)
    
    def load_sample_data(self) -> Tuple[List[str], List[int]]:
        """
        Load or create sample spam/ham data for training
        """
        if os.path.exists(self.data_path):
            # Load existing data
            df = pd.read_csv(self.data_path)
            email_contents = df['email_content'].tolist()
            labels = df['label'].tolist()
        else:
            # Create sample data
            email_contents, labels = self._create_sample_data()
            
            # Save to CSV
            df = pd.DataFrame({
                'email_content': email_contents,
                'label': labels
            })
            df.to_csv(self.data_path, index=False)
        
        return email_contents, labels
    
    def _create_sample_data(self) -> Tuple[List[str], List[int]]:
        """
        Create sample spam and ham emails for training
        """
        spam_emails = [
            "URGENT: You've won $1,000,000! Click here to claim your prize now! Limited time offer!",
            "FREE VIAGRA! Get 50% off on all medications. Buy now before supplies run out!",
            "CONGRATULATIONS! You're the lucky winner of our lottery! Send your bank details to claim $500,000!",
            "Investment opportunity! Double your money in 30 days! Guaranteed returns!",
            "Your account has been suspended. Click here to verify your password immediately!",
            "FREE iPhone 15! You've been selected for our exclusive offer! Claim now!",
            "Make money fast! Work from home and earn $5000 per week! No experience needed!",
            "Your package is delayed. Click here to track and claim compensation!",
            "Bank account verification required. Please provide your credit card details.",
            "Weight loss miracle! Lose 20 pounds in 2 weeks! Order now for 50% discount!",
            "Your computer has been infected with virus! Download our antivirus software now!",
            "FREE gift card! You've been chosen for our promotional offer! Claim your $100 gift card!",
            "Lottery winner! You've won the jackpot! Send your personal information to claim!",
            "Credit card debt relief! We can eliminate your debt! Call now for free consultation!",
            "Your email account will be deleted. Click here to keep your account active!",
            "FREE trial! Get premium software for free! No credit card required!",
            "Your social security number has been compromised. Verify your identity now!",
            "Make money online! Earn $1000 per day working from home! Join our program!",
            "Your Netflix account has been suspended. Click here to reactivate!",
            "FREE cryptocurrency! Get Bitcoin worth $500! Limited time offer!"
        ]
        
        ham_emails = [
            "Hi John, I hope you're doing well. Let's meet for coffee tomorrow at 3 PM.",
            "Dear team, please find attached the quarterly report for your review.",
            "Hello Sarah, thank you for your email. I'll get back to you by Friday.",
            "Meeting reminder: Project review tomorrow at 10 AM in conference room A.",
            "Hi Mom, I'll be home for dinner tonight. See you at 7 PM.",
            "Dear customer, your order #12345 has been shipped and will arrive on Monday.",
            "Hello David, I'm writing to confirm our appointment next Tuesday at 2 PM.",
            "Team update: The new software release is scheduled for next week.",
            "Hi Lisa, thanks for the birthday wishes! I had a great day.",
            "Dear colleagues, please submit your timesheets by Friday.",
            "Hello Mark, I'm available for a call tomorrow between 2-4 PM.",
            "Meeting agenda: Discuss Q4 goals and budget planning.",
            "Hi Tom, the documents you requested are ready for pickup.",
            "Dear family, we're planning a vacation in July. Any suggestions?",
            "Hello team, the office will be closed on Monday for the holiday.",
            "Hi Jane, I've completed the analysis you requested. Here are the results.",
            "Dear client, thank you for choosing our services. We appreciate your business.",
            "Hello everyone, please welcome our new team member starting next week.",
            "Hi Mike, I'm running 15 minutes late for our meeting. Sorry for the inconvenience.",
            "Dear students, the exam schedule has been posted on the website.",
            "Hello friends, I'm organizing a dinner party this weekend. Please RSVP."
        ]
        
        # Combine and shuffle
        all_emails = spam_emails + ham_emails
        all_labels = [1] * len(spam_emails) + [0] * len(ham_emails)
        
        # Shuffle the data
        indices = np.random.permutation(len(all_emails))
        shuffled_emails = [all_emails[i] for i in indices]
        shuffled_labels = [all_labels[i] for i in indices]
        
        return shuffled_emails, shuffled_labels
    
    def train_model(self, save_model: bool = True) -> Dict:
        """
        Train the spam classifier
        """
        print("Loading training data...")
        email_contents, labels = self.load_sample_data()
        print(f"Training on {len(email_contents)} emails ({sum(labels)} spam, {len(labels) - sum(labels)} ham)")
        # Train the model
        print("Training models...")
        performance_metrics = self.classifier.train_all_models(email_contents, labels)
        # Evaluate ensemble performance
        print("[DEBUG] About to call evaluate_performance...")
        ensemble_metrics = self.classifier.evaluate_performance()
        print("[DEBUG] Finished evaluate_performance.")
        # Store results
        self.training_results = {
            'individual_models': performance_metrics,
            'ensemble': ensemble_metrics,
            'training_date': datetime.now().isoformat(),
            'data_size': len(email_contents),
            'spam_count': sum(labels),
            'ham_count': len(labels) - sum(labels)
        }
        # Save model if requested
        if save_model:
            self.save_model()
        # Print results
        self._print_training_results()
        return self.training_results
    
    def save_model(self, filepath: str = 'models/saved/spam_classifier'):
        """
        Save the trained model
        """
        # Save the classifier
        self.classifier.save_models(filepath)
        # Save training results
        def convert_ndarrays_to_lists(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_ndarrays_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_ndarrays_to_lists(i) for i in obj]
            else:
                return obj
        results_filepath = filepath + '_results.json'
        results_to_save = convert_ndarrays_to_lists(self.training_results)
        with open(results_filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"Model saved to {filepath}")
        print(f"Training results saved to {results_filepath}")
    
    def load_model(self, filepath: str = 'models/saved/spam_classifier'):
        """
        Load a trained model
        """
        try:
            self.classifier.load_models(filepath)
            
            # Load training results
            results_filepath = filepath + '_results.json'
            if os.path.exists(results_filepath):
                with open(results_filepath, 'r') as f:
                    self.training_results = json.load(f)
            
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def test_model(self, test_emails: List[str]) -> List[Dict]:
        """
        Test the model on new emails
        """
        if not self.classifier.is_fitted:
            raise ValueError("Model must be trained before testing")
        
        results = self.classifier.predict_batch(test_emails)
        
        for i, (email, result) in enumerate(zip(test_emails, results)):
            print(f"\nEmail {i+1}:")
            print(f"Content: {email[:100]}...")
            print(f"Prediction: {'SPAM' if result['is_spam'] else 'HAM'}")
            print(f"Spam Probability: {result['spam_probability']:.3f}")
            print(f"Confidence: {result['confidence']:.3f}")
        
        return results
    
    def _print_training_results(self):
        """
        Print training results in a formatted way
        """
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        
        # Individual model results
        print("\nIndividual Model Performance:")
        print("-" * 40)
        for model_name, metrics in self.training_results['individual_models'].items():
            print(f"{model_name.upper():20} | "
                  f"Accuracy: {metrics['accuracy']:.4f} | "
                  f"Precision: {metrics['precision']:.4f} | "
                  f"Recall: {metrics['recall']:.4f} | "
                  f"F1: {metrics['f1_score']:.4f}")
        
        # Ensemble results
        ensemble = self.training_results['ensemble']
        print(f"\nEnsemble Performance:")
        print("-" * 40)
        print(f"Accuracy:  {ensemble['accuracy']:.4f}")
        print(f"Precision: {ensemble['precision']:.4f}")
        print(f"Recall:    {ensemble['recall']:.4f}")
        print(f"F1 Score:  {ensemble['f1_score']:.4f}")
        
        # Data summary
        print(f"\nData Summary:")
        print("-" * 40)
        print(f"Total emails: {self.training_results['data_size']}")
        print(f"Spam emails:  {self.training_results['spam_count']}")
        print(f"Ham emails:   {self.training_results['ham_count']}")
        print(f"Training date: {self.training_results['training_date']}")
    
    def create_performance_plots(self, save_plots: bool = True):
        """
        Create performance visualization plots
        """
        if not self.training_results:
            print("No training results available. Train the model first.")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Spam Classifier Performance Analysis', fontsize=16)
        
        # 1. Model comparison
        model_names = list(self.training_results['individual_models'].keys())
        accuracies = [self.training_results['individual_models'][name]['accuracy'] for name in model_names]
        
        axes[0, 0].bar(model_names, accuracies, color='skyblue')
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Precision vs Recall
        precisions = [self.training_results['individual_models'][name]['precision'] for name in model_names]
        recalls = [self.training_results['individual_models'][name]['recall'] for name in model_names]
        
        axes[0, 1].scatter(precisions, recalls, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (precisions[i], recalls[i]), xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. F1 Score comparison
        f1_scores = [self.training_results['individual_models'][name]['f1_score'] for name in model_names]
        
        axes[1, 0].bar(model_names, f1_scores, color='lightgreen')
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Confusion matrix for ensemble
        cm = np.array(self.training_results['ensemble']['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Ensemble Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        axes[1, 1].set_xticklabels(['Ham', 'Spam'])
        axes[1, 1].set_yticklabels(['Ham', 'Spam'])
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('models/saved/performance_plots.png', dpi=300, bbox_inches='tight')
            print("Performance plots saved to models/saved/performance_plots.png")
        
        plt.show()
    
    def get_feature_importance_analysis(self, top_n: int = 20):
        """
        Analyze and display feature importance
        """
        if not self.classifier.is_fitted:
            print("Model must be trained before analyzing feature importance")
            return
        
        importance_dict = self.classifier.get_feature_importance()
        
        if not importance_dict:
            print("No feature importance available for this model")
            return
        
        # Get top features
        top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        print(f"\nTop {top_n} Most Important Features:")
        print("-" * 50)
        for feature, importance in top_features:
            print(f"{feature:30} | {importance:.4f}")
        
        # Create feature importance plot
        features, importances = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), importances, color='orange')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        plt.savefig('models/saved/feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved to models/saved/feature_importance.png")
        plt.show()
    
    def cross_validate_model(self, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation on the model
        """
        print(f"Performing {cv_folds}-fold cross-validation...")
        
        email_contents, labels = self.load_sample_data()
        
        # Use a simpler model for cross-validation (Random Forest)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        # from sklearn.preprocessing import StandardScaler
        
        # Extract features
        # Do NOT use the global classifier or scaler here
        feature_extractor = FeatureExtractor()
        X = feature_extractor.fit_transform(email_contents)
        y = np.array(labels)
        
        # NOTE: No scaling is applied here because RandomForest does not require it.
        # If you want to use a model that requires scaling, fit a StandardScaler on each fold.
        # Example:
        # from sklearn.model_selection import StratifiedKFold
        # skf = StratifiedKFold(n_splits=cv_folds)
        # scores = []
        # for train_idx, test_idx in skf.split(X, y):
        #     scaler = StandardScaler().fit(X[train_idx])
        #     X_train_scaled = scaler.transform(X[train_idx])
        #     X_test_scaled = scaler.transform(X[test_idx])
        #     model = ...
        #     model.fit(X_train_scaled, y[train_idx])
        #     scores.append(model.score(X_test_scaled, y[test_idx]))
        # cv_scores = np.array(scores)
        
        # Perform cross-validation
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(rf_model, X, y, cv=cv_folds, scoring='accuracy')
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_accuracy': cv_scores.mean(),
            'std_accuracy': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        print(f"Cross-validation results:")
        print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f} (+/- {cv_results['std_accuracy'] * 2:.4f})")
        
        return cv_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/spam_data.csv', help='Path to CSV data file')
    args = parser.parse_args()
    try:
        # Create and run the model trainer
        trainer = ModelTrainer(data_path=args.data_path)
        print("Starting training process...")
        print("[DEBUG] Calling train_model...")
        # Train the model
        results = trainer.train_model(save_model=True)
        print("[DEBUG] train_model completed.")
        print("\n🎉 Training process completed!")
    except Exception as e:
        print(f"✗ Training failed: {str(e)}")
        traceback.print_exc()
        sys.exit(1) 