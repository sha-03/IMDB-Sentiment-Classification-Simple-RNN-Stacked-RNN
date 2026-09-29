# SIMPLE RNN + STACKED RNN - IMDB SENTIMENT CLASSIFICATION

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

max_features = 10000
maxlen = 500

# Load data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad reviews
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# ---------------- SIMPLE RNN ----------------
simple_rnn = Sequential([
    Embedding(max_features, 32),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

simple_rnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nSIMPLE RNN")
simple_rnn.summary()

simple_rnn.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

simple_loss, simple_acc = simple_rnn.evaluate(x_test, y_test, verbose=0)
print("Simple RNN Test Accuracy:", simple_acc)

# ---------------- STACKED RNN ----------------
stacked_rnn = Sequential([
    Embedding(max_features, 32),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32),
    Dense(1, activation="sigmoid")
])

stacked_rnn.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nSTACKED RNN")
stacked_rnn.summary()

stacked_rnn.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2
)

test_loss, test_acc = stacked_rnn.evaluate(x_test, y_test, verbose=0)
print("Stacked RNN Test Accuracy:", test_acc)
