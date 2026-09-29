# 🎬 IMDB Sentiment Classification using Simple RNN and Stacked RNN

## 📌 Overview
A Deep Learning project that classifies IMDB movie reviews as **Positive** or **Negative** using **Simple RNN** and **Stacked RNN** models.

## 🎯 Objective
To understand how Recurrent Neural Networks process sequential text and compare a single RNN layer with a deeper stacked RNN architecture.

## 🧠 Architectures

### Simple RNN
`Review → Embedding → SimpleRNN → Dense → Positive/Negative`

### Stacked RNN
`Review → Embedding → SimpleRNN → SimpleRNN → Dense → Positive/Negative`

The first layer of the Stacked RNN uses `return_sequences=True` so its sequence output can be passed to the second RNN layer.

## 📊 Dataset
The project uses the **IMDB Movie Review Dataset** from TensorFlow/Keras.

- Top 10,000 words
- Maximum sequence length: 500 tokens
- Binary classes: Positive and Negative

## ⚙️ Technologies
- Python
- TensorFlow
- Keras
- SimpleRNN
- Embedding
- IMDB Dataset

## 🔄 Workflow
1. Load the IMDB dataset.
2. Keep the top 10,000 words.
3. Pad reviews to 500 tokens.
4. Convert words into vectors with an Embedding layer.
5. Build and train a Simple RNN.
6. Evaluate its test accuracy.
7. Build and train a Stacked RNN.
8. Evaluate its test accuracy.

## ▶️ Run

Install TensorFlow:

```bash
pip install tensorflow
```

Run:

```bash
python imdb_sentiment_rnn.py
```

## 📁 Files

```text
IMDB-Sentiment-RNN/
├── imdb_sentiment_rnn.py
├── README.md
└── requirements.txt
```

## 📚 Learning Outcomes
- RNN fundamentals
- Sequential text processing
- Word embeddings
- Sequence padding
- Binary sentiment classification
- `return_sequences=True`
- Stacked RNN architecture

## 👨‍💻 Author
**Sham Anand**
