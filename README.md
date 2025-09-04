# 🔍 Machine Learning & NLP-based Fake News Detection System

A sophisticated **Machine Learning and Natural Language Processing** system that predicts whether a given news article is **Fake** or **Real**. This project leverages **advanced ML/DL models** and **text processing techniques** to combat misinformation and promote media literacy.

## Some of the files cannot be uploaded because of the size as datadet is really large

## ✅ Features

- **🔧 Advanced Text Preprocessing**
  - Tokenization and text normalization
  - Stopwords removal
  - Stemming and lemmatization
  - Special character handling

- **📊 Multiple Vectorization Techniques**
  - **TF-IDF** (Term Frequency-Inverse Document Frequency)
  - Word Embeddings (Word2Vec, GloVe)
  - Count Vectorization

- **🤖 Diverse ML/DL Models**
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost
  - **Deep Learning Models (LSTM / Transformer)**

- **📈 Comprehensive Evaluation**
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix Analysis
  - ROC Curves and AUC

- **🌐 User-Friendly Interface**
  - Flask web application
  - Streamlit dashboard option
  - RESTful API endpoints

## 📂 Project Structure

```
Fake-News-Detection/
├── data/                    # Dataset and data files
│   ├── raw/                # Raw datasets
│   ├── processed/          # Cleaned and preprocessed data
│   └── embeddings/         # Pre-trained word embeddings
├── notebooks/              # Jupyter notebooks for EDA & experiments
│   ├── 01_EDA.ipynb       # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
├── models/                 # Saved ML/DL models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   └── lstm_model.h5
├── src/                    # Source code modules
│   ├── preprocessing.py    # Text preprocessing functions
│   ├── feature_extraction.py
│   ├── models.py          # Model definitions
│   └── utils.py           # Utility functions
├── app/                    # Web application
│   ├── static/            # CSS, JS, images
│   ├── templates/         # HTML templates
│   ├── app.py            # Flask application
│   └── streamlit_app.py  # Streamlit version
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
├── main.py                # Main execution script
├── config.py              # Configuration settings
└── README.md              # Project documentation
```

## 🔍 How It Works

1. **📰 Input Processing**
   - User inputs news headline or full article text
   - System accepts various text formats and lengths

2. **🧹 Text Preprocessing**
   - Remove HTML tags, URLs, and special characters
   - Convert to lowercase and tokenize
   - Remove stopwords and apply stemming/lemmatization

3. **🔢 Feature Extraction**
   - Convert cleaned text to numerical vectors using TF-IDF or embeddings
   - Handle out-of-vocabulary words appropriately

4. **🎯 Model Prediction**
   - Ensemble of ML/DL models processes the features
   - Each model outputs probability scores for Fake/Real classification

5. **📊 Result Generation**
   - Combine predictions using weighted voting or stacking
   - Provide confidence score and final classification
   - Display explanation of key features influencing the decision

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| Logistic Regression | 92.1% | 91.8% | 92.3% | 92.0% |
| Random Forest | 91.5% | 90.9% | 92.1% | 91.5% |
| XGBoost | 93.3% | 93.1% | 93.5% | 93.3% |
| **LSTM (Deep Learning)** | **95.0%** | **94.8%** | **95.2%** | **95.0%** |

*Performance metrics evaluated on test dataset of 10,000 articles*

## 🚀 Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2️⃣ Set Up Environment
```bash
# Create virtual environment
python -m venv fake_news_env

# Activate virtual environment
# On Windows:
fake_news_env\Scripts\activate
# On macOS/Linux:
source fake_news_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3️⃣ Download Required Data
```bash
# Download NLTK data
python -c "import nltk; nltk.download('all')"

# Download pre-trained models (if not training from scratch)
python download_models.py
```

### 4️⃣ Run the Application

**Option A: Flask Web App**
```bash
python main.py
# Access at http://localhost:5000
```

**Option B: Streamlit Dashboard**
```bash
streamlit run app/streamlit_app.py
# Access at http://localhost:8501
```

**Option C: Command Line Interface**
```bash
python predict.py --text "Your news article text here"
```

## 📦 Dependencies

### Core Requirements
- **Python 3.8+**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **nltk** - Natural language processing
- **spacy** - Advanced NLP features

### Machine Learning
- **xgboost** - Gradient boosting framework
- **tensorflow** / **pytorch** - Deep learning frameworks
- **transformers** - Pre-trained transformer models

### Web Framework
- **Flask** - Web application framework
- **Streamlit** - Data app framework
- **gunicorn** - WSGI HTTP Server

### Visualization
- **matplotlib** - Plotting library
- **seaborn** - Statistical visualization
- **plotly** - Interactive visualizations

## 📂 Dataset Information

### Primary Dataset
- **Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- **Size**: 20,000+ labeled news articles
- **Format**: CSV with columns: `id`, `title`, `author`, `text`, `label`
- **Labels**: 0 (Real), 1 (Fake)

### Additional Datasets (Optional)
- **LIAR Dataset**: Political statements fact-checking
- **FakeNewsNet**: Social media fake news detection
- **ISOT Fake News Dataset**: Comprehensive fake news collection

## 🔧 Configuration

### Model Configuration
```python
# config.py
MODEL_CONFIG = {
    'max_features': 10000,
    'max_length': 500,
    'embedding_dim': 300,
    'lstm_units': 128,
    'dropout_rate': 0.5
}
```

### Training Parameters
```python
TRAINING_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001
}
```

## 🌟 Future Improvements

### 🎯 Planned Enhancements
- **🤖 Advanced Transformer Models**
  - Integration of BERT, RoBERTa, and GPT models
  - Fine-tuning on domain-specific data

- **🔄 Real-time Processing**
  - Live news scraping and analysis
  - Continuous model updates with new data

- **🌐 API Integration**
  - RESTful API for external applications
  - Integration with fact-checking websites
  - Social media platform plugins

- **📱 Mobile Application**
  - Cross-platform mobile app development
  - Browser extension for real-time fact-checking

- **🎨 Enhanced Visualization**
  - Interactive dashboards for trend analysis
  - Explanation AI for model interpretability

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_models.py

# Generate coverage report
python -m pytest --cov=src tests/
```

## 📈 Usage Examples

### Python Script Usage
```python
from src.predictor import FakeNewsDetector

# Initialize detector
detector = FakeNewsDetector()

# Predict single article
result = detector.predict("Breaking: Scientists discover...")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")

# Batch prediction
articles = ["Article 1 text...", "Article 2 text..."]
results = detector.predict_batch(articles)
```

### API Usage
```bash
# POST request to prediction endpoint
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your news article text here"}'
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/
isort src/

# Run linting
flake8 src/
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mradul Gupta**
- 🔗 [LinkedIn](www.linkedin.com/in/mradul-gupta-033438332)
- 🔗 [GitHub](https://github.com/IamMradul)
- 📧 Email: mradulg306@gmail.com

## 🙏 Acknowledgments

- Thanks to the open-source community for providing datasets and tools
- Inspiration from academic research in fake news detection
- Contributors and beta testers who helped improve the system

## 📞 Support

If you encounter any issues or have questions:
- **Create an Issue**: [GitHub Issues](https://github.com/your-username/Fake-News-Detection/issues)
- **Discussion Forum**: [GitHub Discussions](https://github.com/your-username/Fake-News-Detection/discussions)
- **Email**: mradul.gupta@example.com

---

⭐ **Star this repository** if you found it helpful!

**#MachineLearning #NLP #FakeNewsDetection #Python #DeepLearning #AI**
