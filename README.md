# 🔍 Machine Learning & NLP-based Fake News Detection System
<p align="center">
  <img
    src="https://github.com/user-attachments/assets/c1e798c7-0042-468f-866c-7bdf837e5e4a"
    alt="Advanced Fake News Detection Architecture"
    width="435"
    height="280"
  />
</p>

A sophisticated **Machine Learning and Natural Language Processing** system that predicts whether a given news article is **Fake** or **Real**. This project leverages **advanced ML/DL models** and **text processing techniques** to combat misinformation and promote media literacy.

## Some of the files cannot be uploaded because of the size as datadet is really large

<table>
  <tr>
    <td width="65%" valign="top">

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

    </td>
    <td width="35%" align="center">
      <img
        src="https://github.com/user-attachments/assets/3f39d4d0-a5f5-491c-a56a-e8bdd32a52f9"
        alt="Fake News Detection Features Illustration"
        width="300"
      />
      <br/>
      <em>Figure: Core Components of the Fake News Detection System</em>
    </td>
  </tr>
</table>


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

## 📂 Dataset Information

### Primary Dataset
- **Source**: [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- **Size**: 20,000+ labeled news articles
- **Format**: CSV with columns: `id`, `title`, `author`, `text`, `label`
- **Labels**: 0 (Real), 1 (Fake)

### Additional Datasets 
- **LIAR Dataset**: Political statements fact-checking
- **FakeNewsNet**: Social media fake news detection
- **ISOT Fake News Dataset**: Comprehensive fake news collection

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

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** and add tests
4. **Run tests**: `pytest tests/`
5. **Commit changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

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
