# Hate Speech Detection using TensorFlow

This project builds a **multi-class hate speech detection model** using deep learning with TensorFlow. The dataset consists of tweets categorized into three classes:

- **0** → Hate Speech  
- **1** → Offensive Language  
- **2** → Neither  

---

## Project Overview

- **Dataset**: Labeled tweets for hate speech detection.
- **Preprocessing**:
  - Removing URLs, mentions, hashtags, and special characters.
  - Converting text to lowercase.
  - Tokenizing and padding sequences.
- **Model**:
  - **Embedding Layer**: Converts words into dense vectors.
  - **Global Average Pooling**: Reduces dimensionality.
  - **Dense Layers**: Fully connected layers for classification.
  - **Softmax Activation**: Outputs class probabilities.
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix, ROC Curves.

---

## Dataset

The dataset (`labeled_data.csv`) contains:
- `class`: Labels (0 = Hate Speech, 1 = Offensive, 2 = Neither).
- `tweet`: The tweet text.

---

## Installation & Requirements

Ensure you have Python and the required libraries installed:

```bash
pip install tensorflow pandas numpy matplotlib seaborn nltk wordcloud scikit-learn
