# 📰 Fake News Detection Using Random Forest

A machine learning project that classifies news articles as fake or real using natural language processing (NLP) and a Random Forest Classifier. The model achieves over **93% accuracy** on a cleaned and preprocessed dataset.

---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 📖 Introduction

Fake news has become a significant problem in the digital age, particularly in social media and online platforms. This project leverages **Natural Language Processing (NLP)** and **scikit-learn**'s **Random Forest** algorithm to detect fake news based on article content.

---

## ✨ Features

- Clean and preprocess text data using NLTK
- Lemmatization and stemming
- TF-IDF vectorization of combined article title and body
- Train/test split and model evaluation
- Confusion matrix heatmap visualization
- Achieves 93%+ accuracy with Random Forest Classifier

---

## 💾 Installation

Clone the repository and install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure to download necessary NLTK corpora:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

---

## 🚀 Usage

Run the main script after placing your `train.csv` file in the project directory:

```bash
python fake_news_detection.py
```

This script performs:
- Data loading and cleaning
- Text preprocessing
- TF-IDF vectorization
- Model training with `RandomForestClassifier`
- Accuracy evaluation
- Confusion matrix plotting

---

## 📁 Project Structure

```text
.
├── train.csv
├── fake_news_detection.py
├── README.md
├── requirements.txt
└── outputs/
    └── confusion_matrix.png
```

---

## 🧹 Data Preprocessing

- Removed rows with missing `title` or `text`
- Combined `title` and `text` into one field
- Applied:
  - Lowercasing
  - Removing special characters
  - Tokenization
  - Stopword removal
  - Lemmatization and stemming
- Rejoined tokens for vectorization

---

## 🧠 Model Training

- **Vectorizer**: `TfidfVectorizer` from `sklearn`
- **Model**: `RandomForestClassifier`
- **Split**: 70% training, 30% testing
- Achieved **93.37% accuracy** on the test set

---

## 📊 Evaluation

- Accuracy score printed to console
- Confusion matrix visualized with Seaborn heatmap

---

## 🧩 Dependencies

```txt
pandas
numpy
nltk
scikit-learn
matplotlib
seaborn
```

Install them via:

```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```

---

## 🧪 Examples

Example input (`combined` column):

```
LAW ENFORCEMENT ON HIGH ALERT Following Threat... No comment is expected from Barack Obama...
```

Example preprocessed (`clean_text_str`):

```
law enforc high alert follow threat cop white hous
```

---

## 🛠️ Troubleshooting

- **NLTK not working?** Make sure to download required corpora (`punkt`, `stopwords`, `wordnet`)
- **Accuracy too low?** Check for data leakage or try a different classifier
- **Vectorizer memory issues?** Try limiting `max_features` in `TfidfVectorizer`


---
## Contributors
[M.SRIKAR VARDHAN] (https://github.com/M-SRIKAR-VARDHAN)
