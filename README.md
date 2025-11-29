# 💬 Code‑Mixed Sentiment Analysis with BiLSTM (TensorFlow/Keras)

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End‑to‑end pipeline for sentiment classification on code‑mixed (Hinglish) tweets using regex cleaning, Keras Tokenizer, and a BiLSTM classifier.

</div>

---

## 📌 Overview

This repository trains and evaluates a BiLSTM model on Sentimix‑style CSVs (train/val/test) to classify tweets into negative, neutral, and positive sentiment. It includes consistent preprocessing, tokenization, model training, evaluation with classification reports, and simple inference for new texts.

Note: This is a research/education project.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🧹 Preprocessing | Lowercasing; removes URLs, @mentions; converts hashtags to words; compresses elongated chars; strips punctuation and extra spaces. |
| 🔤 Tokenization | Keras Tokenizer with 20k vocabulary and post-padding to 100 tokens; OOV handled via <UNK>. |
| 🧠 Model | Embedding → Bidirectional LSTM(128, return_sequences) → GlobalAveragePooling1D → Dense(64, ReLU) → Dropout(0.3) → Softmax(3). |
| 📊 Evaluation | Classification reports (precision/recall/F1) for validation and test sets; confusion matrices generated in notebook. |
| 🚀 Inference | Clean, tokenize, and predict sentiments for new short texts. |

---

## 📂 Project Structure

```plaintext
sentimix-bilstm/
├── notebook.ipynb                 # Main workflow (training, evaluation, inference)
├── README.md
├── LICENSE
└── data/                          # Place your CSVs here (not in repo)
    ├── sentimix_train.csv
    ├── sentimix_val.csv
    └── sentimix_test.csv
```

---

## 📦 Dataset

- Expected CSV columns:
  - tweet: raw text
  - sentiment: one of {negative, neutral, positive}
- Files used:
  - sentimix_train.csv, sentimix_val.csv, sentimix_test.csv
- Cleaning produces a text_clean column used for tokenization.

Update paths in the notebook if your files live elsewhere (e.g., /content/…).

---

## 🧠 Technical Details

- Vocabulary size: 20,000
- Max sequence length: 100
- Labels (sorted from training): ['negative', 'neutral', 'positive']
- Model
  - Embedding(VOCAB_SIZE=20k, 128) → BiLSTM(128, return_sequences=True)
  - GlobalAveragePooling1D
  - Dense(64, activation="relu") + Dropout(0.3)
  - Dense(3, activation="softmax")
- Training
  - Loss: categorical_crossentropy
  - Optimizer: Adam
  - Metrics: accuracy
  - Epochs: 5
  - Batch size: 64

---

## 🚀 Getting Started

### Installation
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

### Data Preparation
- Place train/val/test CSVs under data/ or adjust the notebook paths:
  - /content/sentimix_train.csv
  - /content/sentimix_val.csv
  - /content/sentimix_test.csv

### Run
- Open notebook.ipynb and execute cells in order:
  1) Load CSVs
  2) Clean text
  3) Tokenize and pad
  4) Encode labels
  5) Build and train BiLSTM
  6) Evaluate on val/test
  7) Run inference on sample texts

---

## 📊 Results

Your run produced:
- Validation accuracy: 0.59
  - Macro F1: 0.59, Weighted F1: 0.59
- Test accuracy: 0.62
  - Macro F1: 0.63, Weighted F1: 0.62

Common error pattern: neutral is the hardest class and gets confused with polar classes.

---



## 🔧 Tips & Next Steps

- Try pretrained embeddings (fastText, GloVe) or character/subword tokenizers.
- Fine‑tune transformer baselines (mBERT, XLM‑R) for code‑mixed text.
- Address class imbalance with class weights or focal loss.
- Add early stopping, learning rate schedules, and more epochs for higher accuracy.
- Evaluate with macro‑F1 as a primary metric for imbalanced multi‑class settings.

---

## 🧪 Reproducibility

- Save artifacts:
  - tokenizer.json
  - label2id.json
  - model.h5 (or SavedModel)
- Fix random seeds and log dataset versions.
- Keep preprocessing identical between train and inference.

---

## 📄 License

Released under the MIT License. See LICENSE.


