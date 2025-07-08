import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import re
from typing import Dict, List, Tuple, Optional
import pickle

from utils.text_preprocessor import TextPreprocessor
from utils.email_parser import EmailParser

class FeatureExtractor:
    """
    Advanced feature extraction for email spam classification
    """
    
    def __init__(self, max_features=1000, ngram_range=(1, 1)):
        """Initialize the feature extractor"""
        self.max_features = max_features
        self.ngram_range = ngram_range
        
        # Text preprocessing
        self.text_preprocessor = TextPreprocessor()
        self.email_parser = EmailParser()
        
        # Vectorizers
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            min_df=1,  # Changed from 2 to 1 for small datasets
            max_df=1.0  # Changed from 0.95 to 1.0 for small datasets
        )
        
        self.count_vectorizer = CountVectorizer(
            max_features=max_features // 2,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,  # Changed from 2 to 1 for small datasets
            max_df=1.0  # Changed from 0.95 to 1.0 for small datasets
        )
        
        # Topic modeling
        self.lda_model = LatentDirichletAllocation(
            n_components=10,
            random_state=42,
            max_iter=10
        )
        
        # LSA n_components will be set dynamically in fit()
        self.lsa_model = None
        
        # Scalers
        self.metadata_scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        
        # Feature names
        self.feature_names = []
        self.metadata_features = []
        
        # Fitted flag
        self.is_fitted = False
    
    def extract_features(self, email_content: str, email_data: Optional[Dict] = None) -> np.ndarray:
        """
        Extract comprehensive features from email content
        """
        # Preprocess text
        processed_text = self.text_preprocessor.preprocess_text(email_content)
        
        # Extract different types of features
        text_features = self._extract_text_features(processed_text).flatten()
        metadata_features = self._extract_metadata_features(email_content, email_data).flatten()
        statistical_features = self._extract_statistical_features(email_content).flatten()
        semantic_features = self._extract_semantic_features(processed_text).flatten()

        # Pad/truncate to expected lengths (set during fit)
        if hasattr(self, 'expected_text_len'):
            if text_features.shape[0] < self.expected_text_len:
                text_features = np.pad(text_features, (0, self.expected_text_len - text_features.shape[0]), mode='constant')
            elif text_features.shape[0] > self.expected_text_len:
                text_features = text_features[:self.expected_text_len]
        if hasattr(self, 'expected_metadata_len'):
            if metadata_features.shape[0] < self.expected_metadata_len:
                metadata_features = np.pad(metadata_features, (0, self.expected_metadata_len - metadata_features.shape[0]), mode='constant')
            elif metadata_features.shape[0] > self.expected_metadata_len:
                metadata_features = metadata_features[:self.expected_metadata_len]
        if hasattr(self, 'expected_stat_len'):
            if statistical_features.shape[0] < self.expected_stat_len:
                statistical_features = np.pad(statistical_features, (0, self.expected_stat_len - statistical_features.shape[0]), mode='constant')
            elif statistical_features.shape[0] > self.expected_stat_len:
                statistical_features = statistical_features[:self.expected_stat_len]
        if hasattr(self, 'expected_semantic_len'):
            if semantic_features.shape[0] < self.expected_semantic_len:
                semantic_features = np.pad(semantic_features, (0, self.expected_semantic_len - semantic_features.shape[0]), mode='constant')
            elif semantic_features.shape[0] > self.expected_semantic_len:
                semantic_features = semantic_features[:self.expected_semantic_len]

        # Combine all features
        all_features = np.concatenate([
            text_features,
            metadata_features,
            statistical_features,
            semantic_features
        ])
        
        return all_features
    
    def _extract_text_features(self, processed_text: str) -> np.ndarray:
        """Extract text-based features"""
        # Only fit_transform during fit phase, otherwise always transform
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before extracting features.")
        tfidf_features = self.tfidf_vectorizer.transform([processed_text]).toarray()
        count_features = self.count_vectorizer.transform([processed_text]).toarray()
        return np.concatenate([tfidf_features, count_features], axis=1)
    
    def _extract_metadata_features(self, email_content: str, email_data: Optional[Dict] = None, scale: bool = True) -> np.ndarray:
        """Extract email metadata features"""
        metadata = {}
        if email_data:
            # Use provided email data
            metadata = self.text_preprocessor.extract_email_metadata(email_content)
            email_summary = self.email_parser.get_email_summary(email_data)
            # Add email parser metadata
            metadata.update({
                'sender_domain_length': len(email_summary.get('sender_domain', '')),
                'recipient_domain_length': len(email_summary.get('recipient_domain', '')),
                'subject_length': email_summary.get('subject_length', 0),
                'has_attachments': int(email_summary.get('has_attachments', False)),
                'attachment_count': email_summary.get('attachment_count', 0),
                'is_suspicious_sender': int(email_summary.get('is_suspicious_sender', False)),
                'spam_score': email_summary.get('spam_score', 0),
                'priority_normal': int(email_summary.get('priority', 'normal') == 'normal'),
                'priority_high': int(email_summary.get('priority', 'normal') == 'high'),
                'priority_low': int(email_summary.get('priority', 'normal') == 'low')
            })
        else:
            # Extract basic metadata from content
            metadata = self.text_preprocessor.extract_email_metadata(email_content)
        # Convert to array
        metadata_array = np.array(list(metadata.values())).reshape(1, -1)
        if scale:
            if not self.is_fitted:
                # For training, fit the scaler (should only happen after collecting all features)
                metadata_array = self.metadata_scaler.fit_transform(metadata_array)
            else:
                # For prediction, transform using fitted scaler
                metadata_array = self.metadata_scaler.transform(metadata_array)
        return metadata_array
    
    def _extract_statistical_features(self, email_content: str, scale: bool = True) -> np.ndarray:
        """Extract statistical features from email content"""
        features = {}
        
        # Basic text statistics
        text_features = self.text_preprocessor.get_text_features(email_content)
        features.update(text_features)
        
        # Additional statistical features
        words = email_content.split()
        if words:
            # Word length statistics
            word_lengths = [len(word) for word in words]
            features['avg_word_length'] = np.mean(word_lengths)
            features['std_word_length'] = np.std(word_lengths)
            features['max_word_length'] = np.max(word_lengths)
            features['min_word_length'] = np.min(word_lengths)
            
            # Character statistics
            chars = list(email_content)
            features['char_count'] = len(chars)
            features['space_count'] = chars.count(' ')
            features['digit_count'] = sum(1 for c in chars if c.isdigit())
            features['uppercase_count'] = sum(1 for c in chars if c.isupper())
            features['lowercase_count'] = sum(1 for c in chars if c.islower())
            features['punctuation_count'] = sum(1 for c in chars if c in '.,!?;:')
            
            # Ratios
            features['space_ratio'] = features['space_count'] / features['char_count'] if features['char_count'] > 0 else 0
            features['digit_ratio'] = features['digit_count'] / features['char_count'] if features['char_count'] > 0 else 0
            features['uppercase_ratio'] = features['uppercase_count'] / features['char_count'] if features['char_count'] > 0 else 0
            features['lowercase_ratio'] = features['lowercase_count'] / features['char_count'] if features['char_count'] > 0 else 0
            features['punctuation_ratio'] = features['punctuation_count'] / features['char_count'] if features['char_count'] > 0 else 0
            
            # Sentence statistics
            sentences = email_content.split('.')
            features['sentence_count'] = len(sentences)
            features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences if s.strip()])
            
            # Unique word statistics
            unique_words = set(words)
            features['unique_word_ratio'] = len(unique_words) / len(words)
            
            # N-gram statistics
            bigrams = self.text_preprocessor.extract_ngrams(email_content, 2)
            trigrams = self.text_preprocessor.extract_ngrams(email_content, 3)
            features['bigram_count'] = len(bigrams)
            features['trigram_count'] = len(trigrams)
            features['bigram_ratio'] = len(bigrams) / len(words) if words else 0
            features['trigram_ratio'] = len(trigrams) / len(words) if words else 0
        else:
            # Default values for empty content
            features.update({
                'avg_word_length': 0, 'std_word_length': 0, 'max_word_length': 0, 'min_word_length': 0,
                'char_count': 0, 'space_count': 0, 'digit_count': 0, 'uppercase_count': 0,
                'lowercase_count': 0, 'punctuation_count': 0, 'space_ratio': 0, 'digit_ratio': 0,
                'uppercase_ratio': 0, 'lowercase_ratio': 0, 'punctuation_ratio': 0,
                'sentence_count': 0, 'avg_sentence_length': 0, 'unique_word_ratio': 0,
                'bigram_count': 0, 'trigram_count': 0, 'bigram_ratio': 0, 'trigram_ratio': 0
            })
        
        # Convert to array
        statistical_array = np.array(list(features.values())).reshape(1, -1)
        if scale:
            if not self.is_fitted:
                # For training, fit the scaler (should only happen after collecting all features)
                statistical_array = self.feature_scaler.fit_transform(statistical_array)
            else:
                # For prediction, transform using fitted scaler
                statistical_array = self.feature_scaler.transform(statistical_array)
        return statistical_array
    
    def _extract_semantic_features(self, processed_text: str) -> np.ndarray:
        """Extract semantic features using topic modeling"""
        # Determine expected semantic features length
        if self.lsa_model is not None:
            n_components = self.lsa_model.n_components
        else:
            # During training, use a default value
            n_components = 50
        
        if not processed_text.strip():
            # Return zeros for empty text
            return np.zeros((1, 10 + n_components))  # 10 LDA + LSA features
        
        # Prepare text for topic modeling
        text_for_topics = [processed_text]
        
        try:
            count_matrix = self.count_vectorizer.transform(text_for_topics)
            lda_features = self.lda_model.transform(count_matrix)
            
            # LSA robust handling
            n_features = count_matrix.shape[1]
            if self.lsa_model is not None and n_features >= n_components:
                lsa_features = self.lsa_model.transform(count_matrix)
            elif n_features > 0:
                # Fit a temporary SVD with n_features components, pad to n_components
                temp_svd = TruncatedSVD(n_components=min(n_features, n_components), random_state=42)
                temp_svd.fit(count_matrix)
                lsa_partial = temp_svd.transform(count_matrix)
                lsa_features = np.zeros((1, n_components))
                lsa_features[0, :lsa_partial.shape[1]] = lsa_partial[0]
            else:
                lsa_features = np.zeros((1, n_components))
            
            # Combine topic features
            semantic_features = np.concatenate([lda_features, lsa_features], axis=1)
            
        except Exception as e:
            # Fallback to zeros if any error occurs
            print(f"Warning: Error in semantic feature extraction: {e}")
            semantic_features = np.zeros((1, 10 + n_components))
        
        return semantic_features
    
    def fit(self, email_contents: List[str], email_data_list: Optional[List[Dict]] = None):
        """Fit the feature extractor on training data"""
        processed_texts = [self.text_preprocessor.preprocess_text(content) for content in email_contents]
        
        # Fit vectorizers on all processed texts
        self.tfidf_vectorizer.fit(processed_texts)
        self.count_vectorizer.fit(processed_texts)
        self.is_fitted = True  # Set immediately after fitting vectorizers
        
        # Fit topic models
        count_matrix = self.count_vectorizer.transform(processed_texts)
        n_features = count_matrix.shape[1]
        n_components = min(50, n_features) if n_features > 0 else 1
        self.lsa_model = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )
        
        # Only fit models if we have features
        if n_features > 0:
            self.lda_model.fit(count_matrix)
            self.lsa_model.fit(count_matrix)
        else:
            # Create dummy models for empty datasets
            print("Warning: No features found in count matrix, using dummy models")
            self.lsa_model = TruncatedSVD(n_components=1, random_state=42)
            self.lsa_model.fit(np.zeros((1, 1)))
        
        # Determine expected lengths
        expected_tfidf_len = len(self.tfidf_vectorizer.get_feature_names_out())
        expected_count_len = len(self.count_vectorizer.get_feature_names_out())
        expected_text_len = expected_tfidf_len + expected_count_len
        # Collect all metadata features for scaler fitting
        metadata_features_raw = [
            self._extract_metadata_features(email_contents[i], email_data_list[i] if email_data_list else None, scale=False).flatten()
            for i in range(len(email_contents))
        ]
        metadata_features_raw = np.vstack(metadata_features_raw)
        self.metadata_scaler.fit(metadata_features_raw)
        # Now get one sample (scaled) for expected_metadata_len
        temp_metadata = self._extract_metadata_features(email_contents[0], None, scale=True).flatten()
        expected_metadata_len = temp_metadata.shape[0]
        # Collect all statistical features for scaler fitting
        statistical_features_raw = [
            self._extract_statistical_features(email_contents[i], scale=False).flatten()
            for i in range(len(email_contents))
        ]
        statistical_features_raw = np.vstack(statistical_features_raw)
        self.feature_scaler.fit(statistical_features_raw)
        # Now get one sample (scaled) for expected_stat_len
        temp_stat = self._extract_statistical_features(email_contents[0], scale=True).flatten()
        expected_stat_len = temp_stat.shape[0]
        expected_semantic_len = 10 + n_components
        text_features_list = []
        metadata_features_list = []
        statistical_features_list = []
        semantic_features_list = []
        for i, processed_text in enumerate(processed_texts):
            email_data = email_data_list[i] if email_data_list else None
            # Text features
            tfidf = self.tfidf_vectorizer.transform([processed_text]).toarray().flatten()
            count = self.count_vectorizer.transform([processed_text]).toarray().flatten()
            text_features = np.concatenate([tfidf, count], axis=0)
            if text_features.shape[0] < expected_text_len:
                text_features = np.pad(text_features, (0, expected_text_len - text_features.shape[0]), mode='constant')
            elif text_features.shape[0] > expected_text_len:
                text_features = text_features[:expected_text_len]
            text_features = text_features.flatten()
            text_features_list.append(text_features)
            # Metadata (now scaled)
            metadata_features = self._extract_metadata_features(email_contents[i], email_data, scale=True).flatten()
            if metadata_features.shape[0] < expected_metadata_len:
                metadata_features = np.pad(metadata_features, (0, expected_metadata_len - metadata_features.shape[0]), mode='constant')
            elif metadata_features.shape[0] > expected_metadata_len:
                metadata_features = metadata_features[:expected_metadata_len]
            metadata_features = metadata_features.flatten()
            metadata_features_list.append(metadata_features)
            # Statistical (now scaled)
            statistical_features = self._extract_statistical_features(email_contents[i], scale=True).flatten()
            if statistical_features.shape[0] < expected_stat_len:
                statistical_features = np.pad(statistical_features, (0, expected_stat_len - statistical_features.shape[0]), mode='constant')
            elif statistical_features.shape[0] > expected_stat_len:
                statistical_features = statistical_features[:expected_stat_len]
            statistical_features = statistical_features.flatten()
            statistical_features_list.append(statistical_features)
            # Semantic
            semantic_features = self._extract_semantic_features(processed_text)
            if semantic_features.shape[1] < expected_semantic_len:
                pad_width = expected_semantic_len - semantic_features.shape[1]
                semantic_features = np.pad(semantic_features, ((0,0),(0,pad_width)), mode='constant')
            elif semantic_features.shape[1] > expected_semantic_len:
                semantic_features = semantic_features[:, :expected_semantic_len]
            semantic_features = semantic_features.flatten()
            semantic_features_list.append(semantic_features)
        
        # Final bulletproof padding/truncation for all feature types
        max_text_len = max(f.shape[0] for f in text_features_list)
        max_meta_len = max(f.shape[0] for f in metadata_features_list)
        max_stat_len = max(f.shape[0] for f in statistical_features_list)
        max_sem_len = max(f.shape[0] for f in semantic_features_list)
        
        print(f"Debug - Feature dimensions: text={max_text_len}, meta={max_meta_len}, stat={max_stat_len}, sem={max_sem_len}")
        
        for i in range(len(email_contents)):
            if text_features_list[i].shape[0] != max_text_len:
                text_features_list[i] = np.pad(text_features_list[i], (0, max_text_len - text_features_list[i].shape[0]), mode='constant')[:max_text_len]
            if metadata_features_list[i].shape[0] != max_meta_len:
                metadata_features_list[i] = np.pad(metadata_features_list[i], (0, max_meta_len - metadata_features_list[i].shape[0]), mode='constant')[:max_meta_len]
            if statistical_features_list[i].shape[0] != max_stat_len:
                statistical_features_list[i] = np.pad(statistical_features_list[i], (0, max_stat_len - statistical_features_list[i].shape[0]), mode='constant')[:max_stat_len]
            if semantic_features_list[i].shape[0] != max_sem_len:
                semantic_features_list[i] = np.pad(semantic_features_list[i], (0, max_sem_len - semantic_features_list[i].shape[0]), mode='constant')[:max_sem_len]
        
        # Combine all features
        all_features = [
            np.concatenate([
                text_features_list[i],
                metadata_features_list[i],
                statistical_features_list[i],
                semantic_features_list[i]
            ], axis=0)
            for i in range(len(email_contents))
        ]
        all_features_array = np.vstack(all_features)
        
        metadata_start = max_text_len
        metadata_end = metadata_start + max_meta_len
        
        metadata_features = all_features_array[:, metadata_start:metadata_end]
        statistical_features = all_features_array[:, metadata_end:metadata_end+max_stat_len]
        
        self.metadata_scaler.fit(metadata_features)
        self.feature_scaler.fit(statistical_features)
        
        # Store feature names
        self.feature_names = (
            list(self.tfidf_vectorizer.get_feature_names_out()) +
            list(self.count_vectorizer.get_feature_names_out()) +
            ['metadata_' + str(i) for i in range(max_meta_len)] +
            ['statistical_' + str(i) for i in range(max_stat_len)] +
            ['lda_' + str(i) for i in range(10)] +
            ['lsa_' + str(i) for i in range(n_components)]
        )
        
        self.is_fitted = True
        
        # Store expected lengths
        self.expected_text_len = expected_text_len
        self.expected_metadata_len = expected_metadata_len
        self.expected_stat_len = expected_stat_len
        self.expected_semantic_len = expected_semantic_len
        
        return self
    
    def transform(self, email_contents: List[str], email_data_list: Optional[List[Dict]] = None) -> np.ndarray:
        """Transform email contents to feature matrix"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before transform")
        
        features_list = []
        for i, content in enumerate(email_contents):
            email_data = email_data_list[i] if email_data_list else None
            features = self.extract_features(content, email_data)
            features_list.append(features.flatten())
        
        return np.vstack(features_list)
    
    def fit_transform(self, email_contents: List[str], email_data_list: Optional[List[Dict]] = None) -> np.ndarray:
        """Fit the extractor and transform the data"""
        self.fit(email_contents, email_data_list)
        return self.transform(email_contents, email_data_list)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names"""
        return self.feature_names
    
    def save(self, filepath: str):
        """Save the fitted feature extractor"""
        if not self.is_fitted:
            raise ValueError("Feature extractor must be fitted before saving")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """Load a fitted feature extractor"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_non_negative_features(self, email_content: str) -> np.ndarray:
        """
        Extract only non-negative features (TF-IDF + CountVectorizer) for MultinomialNB
        """
        processed_text = self.text_preprocessor.preprocess_text(email_content)
        tfidf = self.tfidf_vectorizer.transform([processed_text]).toarray().flatten()
        count = self.count_vectorizer.transform([processed_text]).toarray().flatten()
        text_features = np.concatenate([tfidf, count], axis=0)
        # Pad/truncate to expected length
        if hasattr(self, 'expected_text_len'):
            if text_features.shape[0] < self.expected_text_len:
                text_features = np.pad(text_features, (0, self.expected_text_len - text_features.shape[0]), mode='constant')
            elif text_features.shape[0] > self.expected_text_len:
                text_features = text_features[:self.expected_text_len]
        return text_features

    def get_non_negative_feature_matrix(self, email_contents: list) -> np.ndarray:
        """
        Extract non-negative feature matrix for a list of emails
        """
        return np.vstack([self.get_non_negative_features(content) for content in email_contents]) 