import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from bs4 import BeautifulSoup
import email
from email import policy
import html
import unicodedata

class TextPreprocessor:
    """
    Advanced text preprocessing for email spam classification
    """
    
    def __init__(self):
        """Initialize the text preprocessor with NLTK components"""
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        try:
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
        
        self.stemmer = PorterStemmer()
        
        # Common spam indicators
        self.spam_indicators = {
            'urgent', 'free', 'money', 'cash', 'winner', 'prize', 'lottery',
            'click', 'buy', 'offer', 'limited', 'act now', 'guaranteed',
            'viagra', 'cialis', 'weight loss', 'diet', 'investment',
            'credit card', 'bank', 'account', 'password', 'verify'
        }
        
        # Email-specific patterns
        self.email_patterns = {
            'urls': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            'currency': r'\$\d+(?:\.\d{2})?',
            'numbers': r'\d+',
            'special_chars': r'[^\w\s]'
        }
    
    def extract_email_metadata(self, email_content):
        """
        Extract metadata from email content
        """
        metadata = {
            'has_links': 0,
            'has_emails': 0,
            'has_phone': 0,
            'has_currency': 0,
            'has_numbers': 0,
            'has_caps_ratio': 0,
            'has_spam_words': 0,
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0
        }
        
        if not email_content:
            return metadata
        
        # Count patterns
        metadata['has_links'] = len(re.findall(self.email_patterns['urls'], email_content))
        metadata['has_emails'] = len(re.findall(self.email_patterns['emails'], email_content))
        metadata['has_phone'] = len(re.findall(self.email_patterns['phone'], email_content))
        metadata['has_currency'] = len(re.findall(self.email_patterns['currency'], email_content))
        metadata['has_numbers'] = len(re.findall(self.email_patterns['numbers'], email_content))
        
        # Text statistics
        words = email_content.split()
        metadata['word_count'] = len(words)
        metadata['char_count'] = len(email_content)
        
        if words:
            metadata['avg_word_length'] = sum(len(word) for word in words) / len(words)
            
            # Caps ratio
            caps_count = sum(1 for char in email_content if char.isupper())
            metadata['has_caps_ratio'] = caps_count / len(email_content) if email_content else 0
            
            # Spam words count
            spam_word_count = sum(1 for word in words if word.lower() in self.spam_indicators)
            metadata['has_spam_words'] = spam_word_count
        
        return metadata
    
    def clean_html(self, text):
        """Remove HTML tags and entities"""
        if not text:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text
    
    def normalize_unicode(self, text):
        """Normalize unicode characters"""
        if not text:
            return ""
        return unicodedata.normalize('NFKD', text)
    
    def remove_urls(self, text):
        """Remove URLs from text"""
        if not text:
            return ""
        return re.sub(self.email_patterns['urls'], 'URL', text)
    
    def remove_emails(self, text):
        """Remove email addresses from text"""
        if not text:
            return ""
        return re.sub(self.email_patterns['emails'], 'EMAIL', text)
    
    def remove_phone_numbers(self, text):
        """Remove phone numbers from text"""
        if not text:
            return ""
        return re.sub(self.email_patterns['phone'], 'PHONE', text)
    
    def remove_currency(self, text):
        """Remove currency symbols and amounts"""
        if not text:
            return ""
        return re.sub(self.email_patterns['currency'], 'MONEY', text)
    
    def remove_numbers(self, text):
        """Remove standalone numbers"""
        if not text:
            return ""
        return re.sub(r'\b\d+\b', 'NUMBER', text)
    
    def remove_punctuation(self, text):
        """Remove punctuation marks"""
        if not text:
            return ""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def remove_extra_whitespace(self, text):
        """Remove extra whitespace and normalize"""
        if not text:
            return ""
        return ' '.join(text.split())
    
    def to_lowercase(self, text):
        """Convert text to lowercase"""
        if not text:
            return ""
        return text.lower()
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        if not text:
            return ""
        try:
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
        except LookupError:
            # Fallback to simple word splitting if NLTK data is not available
            words = text.split()
            filtered_words = [word for word in words if word.lower() not in self.stop_words]
            return ' '.join(filtered_words)
    
    def lemmatize_text(self, text):
        """Lemmatize words in text"""
        if not text:
            return ""
        try:
            words = word_tokenize(text)
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
        except LookupError:
            # Fallback to simple word splitting if NLTK data is not available
            words = text.split()
            lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
            return ' '.join(lemmatized_words)
    
    def stem_text(self, text):
        """Stem words in text"""
        if not text:
            return ""
        try:
            words = word_tokenize(text)
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
        except LookupError:
            # Fallback to simple word splitting if NLTK data is not available
            words = text.split()
            stemmed_words = [self.stemmer.stem(word) for word in words]
            return ' '.join(stemmed_words)
    
    def extract_ngrams(self, text, n=2):
        """Extract n-grams from text"""
        if not text:
            return []
        try:
            words = word_tokenize(text)
        except LookupError:
            # Fallback to simple word splitting if NLTK data is not available
            words = text.split()
        ngrams = []
        for i in range(len(words) - n + 1):
            ngrams.append(' '.join(words[i:i+n]))
        return ngrams
    
    def preprocess_text(self, text, remove_stopwords=True, lemmatize=True, stem=False):
        """
        Complete text preprocessing pipeline
        """
        if not text:
            return ""
        
        # Clean HTML
        text = self.clean_html(text)
        
        # Normalize unicode
        text = self.normalize_unicode(text)
        
        # Remove specific patterns
        text = self.remove_urls(text)
        text = self.remove_emails(text)
        text = self.remove_phone_numbers(text)
        text = self.remove_currency(text)
        text = self.remove_numbers(text)
        
        # Basic cleaning
        text = self.to_lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_extra_whitespace(text)
        
        # Advanced processing
        if remove_stopwords:
            text = self.remove_stopwords(text)
        
        if lemmatize:
            text = self.lemmatize_text(text)
        elif stem:
            text = self.stem_text(text)
        
        return text
    
    def get_text_features(self, text):
        """
        Extract comprehensive text features
        """
        features = {}
        
        if not text:
            return features
        
        # Basic text features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text.replace(' ', ''))
        
        # Sentence features
        try:
            sentences = sent_tokenize(text)
        except LookupError:
            # Fallback to simple sentence splitting if NLTK data is not available
            sentences = text.split('.')
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Word features
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(w) for w in words) / len(words)
            features['unique_words'] = len(set(words))
            features['lexical_diversity'] = len(set(words)) / len(words)
        
        # Spam indicators
        spam_word_count = sum(1 for word in words if word.lower() in self.spam_indicators)
        features['spam_word_ratio'] = spam_word_count / len(words) if words else 0
        
        # N-gram features
        bigrams = self.extract_ngrams(text, 2)
        trigrams = self.extract_ngrams(text, 3)
        features['bigram_count'] = len(bigrams)
        features['trigram_count'] = len(trigrams)
        
        return features 