# Yelp Review Sentiment Analysis using Bidirectional LSTM

A deep learning project that classifies Yelp business reviews as **Positive** or **Negative** using a Bidirectional Long Short-Term Memory (Bi-LSTM) neural network built with TensorFlow/Keras.


---

## Overview

This project applies a Bidirectional LSTM model to binary sentiment classification on the Yelp Polarity dataset. Unlike standard unidirectional LSTMs, the Bi-LSTM processes each review both forward and backward simultaneously, capturing richer contextual relationships between words. A `SpatialDropout1D` layer and `EarlyStopping` callback are included to reduce overfitting on Yelp's large and noisy user-generated text.

---

## Dataset

**Yelp Review Polarity Dataset**

| Property | Value |
|---|---|
| Source | Yelp Open Dataset / Kaggle |
| Training samples | ~560,000 |
| Test samples | ~38,000 |
| Labels | `0` = Negative (1–2 stars), `1` = Positive (4–5 stars) |
| Input | Review body text |

Download options:
- [Kaggle – Yelp Review Polarity](https://www.kaggle.com/datasets/irustam/yelp-reviews-polarity)
- [HuggingFace datasets](https://huggingface.co/datasets/yelp_polarity): `load_dataset('yelp_polarity')`

---

## Model Architecture

```
Embedding(15000, 128, input_length=150)
    ↓
SpatialDropout1D(0.3)
    ↓
Bidirectional(LSTM(64))   →  128-dimensional output
    ↓
Dropout(0.5)
    ↓
Dense(1, activation='sigmoid')
```

| Layer | Output Shape | Parameters |
|---|---|---|
| Embedding | (None, 150, 128) | 1,920,000 |
| SpatialDropout1D | (None, 150, 128) | 0 |
| Bidirectional LSTM | (None, 128) | 131,584 |
| Dropout | (None, 128) | 0 |
| Dense | (None, 1) | 129 |
| **Total** | | **2,051,713** |

---

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `VOCAB_SIZE` | 15,000 | Top N most frequent words retained |
| `MAX_LENGTH` | 150 | Words per review (pad/truncate) |
| `EMBED_DIM` | 128 | Word embedding dimensions |
| `LSTM_UNITS` | 64 | Units per direction (128 total) |
| `BATCH_SIZE` | 64 | Reviews per gradient update |
| `EPOCHS` | 10 | Max training cycles |
| `DROPOUT` | 0.5 | Fraction of neurons deactivated |
| `SPATIAL_DROPOUT` | 0.3 | Fraction of embedding vectors dropped |

---

## Project Structure

```
BiLSTM-Yelp-Sentiment/
│
├── YelpSentimentAnalysis.ipynb   # Main notebook (data → train → evaluate)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
└── outputs/
    ├── training_history.png      # Accuracy & loss curves
    └── model_summary.txt         # Model architecture summary
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Shreejal172/BiLSTM-Yelp-Sentiment.git
cd BiLSTM-Yelp-Sentiment
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

**Option A – HuggingFace (recommended, no manual download):**
```python
from datasets import load_dataset
dataset = load_dataset('yelp_polarity')
```

**Option B – Kaggle CLI:**
```bash
kaggle datasets download -d irustam/yelp-reviews-polarity
unzip yelp-reviews-polarity.zip
```

### 4. Run the notebook

```bash
jupyter notebook YelpSentimentAnalysis.ipynb
```

The notebook contains a built-in demonstration dataset (Option C) so it runs immediately even without downloading the full dataset.

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~92–94% (on full dataset) |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |

Training stops automatically via `EarlyStopping(patience=3)` when validation loss stops improving.

---

## Sample Predictions

```
Review    : The food was absolutely incredible, best meal I had in years!
Sentiment : Positive  (confidence: 0.9741)

Review    : Horrible experience, staff ignored us and food was cold.
Sentiment : Negative  (confidence: 0.9512)

Review    : Decent place, nothing special but not bad either.
Sentiment : Negative  (confidence: 0.5103)   ← model uncertain on neutral text

Review    : I will never return here. Worst service ever experienced.
Sentiment : Negative  (confidence: 0.9887)

Review    : Loved the ambiance and the pasta. Highly recommend!
Sentiment : Positive  (confidence: 0.9634)
```

---

## Requirements

```
tensorflow>=2.10.0
scikit-learn>=1.1.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
datasets>=2.0.0
```

---

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, *9*(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, *45*(11), 2673–2681. https://doi.org/10.1109/78.650093
- Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. *Advances in Neural Information Processing Systems*, *28*, 649–657.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, *15*(1), 1929–1958.

---

## License

This project is submitted as academic coursework for TECH 405 at Presidential Graduate School. For educational use only.
