<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

End‑to‑end pipeline for sentiment classification on Hinglish/code‑mixed tweets using regex cleaning, Keras Tokenizer, and a BiLSTM classifier.

</div>

---

## 📌 Overview
This repo trains and evaluates a BiLSTM model on the Sentimix‑style CSVs (train/val/test). It includes:
- Text cleaning (URLs, mentions, hashtags, repeated chars, punctuation)
- Tokenization + padding (vocab=20k, max_len=100)
- BiLSTM(128) → GAP → Dense(64) → Dropout → Softmax(3)
- Evaluation with classification reports and confusion matrices
- Simple inference for new texts

Note: Educational/research use only.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| 🧹 Robust Preprocessing | Regex cleaning for noisy social text (links, @mentions, #hashtags, elongations). |
| 🔤 Tokenization | Keras Tokenizer (20k vocab), padded to 100 tokens. |
| 🧠 BiLSTM Model | Bidirectional LSTM (128) with global avg pooling and dropout. |
| 📊 Evaluation | Per‑split classification reports; accuracy, macro‑/weighted‑F1. |
| 🚀 Inference | Predict sentiments for new short texts with the trained model. |

---

## 📂 Project Structure

```plaintext
sentimix-bilstm/
├── notebook.ipynb                 # Main workflow (training, eval, inference)
├── README.md
├── LICENSE
└── data/                          # Not in repo; place your CSVs here
    ├── sentimix_train.csv
    ├── sentimix_val.csv
    └── sentimix_test.csv
```

📦 Dataset
Expected CSV columns:
tweet: raw text
sentiment: one of {negative, neutral, positive}
Files used:
sentimix_train.csv, sentimix_val.csv, sentimix_test.csv
Cleaning generates a text_clean column used for tokenization.
Update paths as needed if your files live elsewhere (e.g., /content/…).

🧠 Technical Details
Vocab size: 20,000
Max sequence length: 100
Labels: ['negative', 'neutral', 'positive']
Model:
Embedding(20k, 128) → BiLSTM(128, return_sequences=True)
GlobalAveragePooling1D
Dense(64, relu) + Dropout(0.3)
Dense(3, softmax)
Training: epochs=5, batch_size=64, loss=categorical_crossentropy, optimizer=Adam
🚀 Getting Started
Installation
Bash

pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
Run
Open the notebook and execute cells end‑to‑end.
Ensure CSVs exist at the configured paths.
📊 Results (Your Run)
Validation accuracy: 0.59
Test accuracy: 0.62
Validation macro F1: 0.59 | Weighted F1: 0.59
Test macro F1: 0.63 | Weighted F1: 0.62
Notes:

Neutral is typically hardest; confusion often occurs between neutral and the polar classes.
Results may vary with random seeds and preprocessing choices.
🧪 Inference (Example)
Python

texts = [
    "Yaar aaj mood bohot kharab hai 😞",
    "Party mast thi kal, full enjoy kiya!",
    "@friend tu bohot help kar raha hai, thanks!"
]
# Clean → tokenize → pad with the same tokenizer and MAX_LEN
# model.predict(...) → argmax → map using id2label
Ensure you reuse the exact cleaning, tokenizer, and label mapping created during training.

⚖️ Limitations & Next Steps
No pretrained embeddings; try fastText, GloVe, or subword tokenization.
Consider transformer baselines (XLM‑R, mBERT) for code‑mixed text.
Class imbalance and sarcasm/irony remain challenging.
Add attention, class weighting, or focal loss for potential gains.

🧪 Reproducibility
Save artifacts:
tokenizer.json, label2id.json, model.h5
Fix seeds and log dataset versions.
Keep preprocessing consistent between train and inference.
📄 License
Released under the MIT License. See LICENSE.
⭐️ If this helps your work, a star is appreciated!
"""
