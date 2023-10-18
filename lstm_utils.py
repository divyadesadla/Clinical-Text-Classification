from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import spacy
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class LSTMTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, nlp, max_sequence_length):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.nlp = nlp
        self.max_sequence_length = max_sequence_length
        self.model = self.build_model()

    def build_model(self):
        """
        Build the LSTM model with spaCy word embeddings.
        """
        # Create a limited vocabulary from spaCy's vocab
        limited_vocab = [word.text for word in self.nlp.vocab if word.has_vector and word.is_alpha]
        limited_vocab_size = len(limited_vocab)

        # Print the limited vocabulary size
        print(f"Vocabulary size from spaCy model: {limited_vocab_size}")

        model = Sequential()
        model.add(Embedding(
            input_dim=limited_vocab_size,  # Limited vocabulary size
            output_dim=self.nlp.vocab.vectors_length,  # Dimension of spaCy word embeddings
            weights=[np.array([self.nlp.vocab.get_vector(word) for word in limited_vocab])],  # Use spaCy word embeddings for limited_vocab
            input_length=self.max_sequence_length  # Input sequence length
        ))

        # Print the shape of the embedding layer weights
        print(f"Shape of Keras Embedding layer weights: {model.layers[0].get_weights()[0].shape}")

        model.add(LSTM(128, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(len(np.unique(self.y_train)), activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def train_model(self, epochs=10, batch_size=32):
        """
        Train the LSTM model with spaCy word embeddings.
        """
        X_train_sequences = self.vectorize_text(self.X_train)
        X_test_sequences = self.vectorize_text(self.X_test)

        history = self.model.fit(
            X_train_sequences, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_sequences, self.y_test),
            verbose=2
        )
        return history

    def vectorize_text(self, text_data):
        """
        Convert text data into sequences of word indices using spaCy word embeddings.
        """
        sequences = []
        for text in text_data:
            doc = self.nlp(text)
            word_indices = [token.rank for token in doc if token.has_vector]
            sequences.append(word_indices)

        return pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')


