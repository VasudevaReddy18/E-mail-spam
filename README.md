# Advanced Email Spam Classifier

An advanced machine learning-based email spam classifier with multiple algorithms, feature engineering, and a web interface.

## Features

- **Multiple ML Algorithms**: Naive Bayes, SVM, Random Forest, Neural Networks
- **Advanced Feature Engineering**: TF-IDF, word embeddings, email metadata
- **Web Interface**: User-friendly Flask-based UI
- **RESTful API**: Easy integration with other applications
- **Comprehensive Evaluation**: Cross-validation, confusion matrix, performance metrics
- **Real-time Processing**: Instant spam detection
- **Model Persistence**: Save and load trained models

## Project Structure

```
├── app.py                 # Main Flask application
├── models/               # Machine learning models
│   ├── spam_classifier.py
│   ├── feature_extractor.py
│   └── model_trainer.py
├── data/                 # Data and datasets
│   ├── spam_data.csv
│   └── sample_emails/
├── static/              # Web assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/           # HTML templates
├── utils/              # Utility functions
│   ├── email_parser.py
│   ├── text_preprocessor.py
│   └── evaluation.py
├── notebooks/          # Jupyter notebooks for analysis
└── tests/             # Unit tests
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## Usage

### Web Interface
```bash
python app.py
```
Visit `http://localhost:5000` to use the web interface.

### API Usage
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"email_content": "Your email content here"}'
```

### Training Models
```python
from models.model_trainer import ModelTrainer

trainer = ModelTrainer()
trainer.train_all_models()
```



## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License 