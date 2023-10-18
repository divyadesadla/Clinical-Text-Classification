import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from utils import DataPreprocessor, LSTMTrainer, ModelOutput
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
import spacy

# Load the dataset
data_preprocessor = DataPreprocessor('mtsamples.csv')
model_data = data_preprocessor.clean_and_preprocess_text_lstm()
model_data = model_data.reset_index(drop=True)
X = model_data['transcription']
y = model_data['medical_specialty']

## Look at distributions of words and sentences ##
# Calculate unique words 
words = [word for text in X for word in word_tokenize(text)]
word_freq_dist = nltk.FreqDist(words)
unique_words = len(word_freq_dist)
print(f'Number of Unique Words: {unique_words}')

# Calculate the sentence length distribution
sentence_lengths = [len(word_tokenize(text)) for text in X]
max_seq_length = max(sentence_lengths)
min_seq_length = min(sentence_lengths)
avg_seq_length = sum(sentence_lengths) / len(sentence_lengths)
print(f'Maximum Sequence Length: {max_seq_length}')
print(f'Minimum Sequence Length: {min_seq_length}')
print(f'Average Sequence Length: {avg_seq_length}')

# Define the vocabulary size and maximum sequence length
max_sequence_length = 200  # Can be fine tuned further

# Load spaCy model for word embeddings
nlp = spacy.load("en_core_web_md")

# Get the vocabulary size
vocab_size = len(nlp.vocab)
print(f"Vocabulary size: {vocab_size}")

# Encoding labels
encoded_labels, label_mapping = data_preprocessor.encode_labels(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, test_size=0.2, random_state=42)

# Limit the vocabulary to common words
limited_vocab = [word.text for word in nlp.vocab if word.has_vector and word.is_alpha]
nlp.vocab.vectors.name = "custom_vectors"  # Change the name of vectors to avoid conflicts

# Filter out text data based on the limited vocabulary
X_train = [text for text in X_train if all(word in limited_vocab for word in text.split())]
X_test = [text for text in X_test if all(word in limited_vocab for word in text.split())]

# Create an instance of LSTMTrainer with the loaded spaCy model
lstm_trainer = LSTMTrainer(X_train, X_test, y_train, y_test, nlp, max_sequence_length)

# Train the model
history = lstm_trainer.train_model(epochs=10, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = lstm_trainer.model.evaluate(lstm_trainer.X_test, lstm_trainer.y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

