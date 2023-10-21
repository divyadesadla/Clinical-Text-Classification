"""
This code is just for exploratory purposes, you don't need to run it!
In this repository, I plan to explore the following methods to classify clinical text:

1. ML Text Classification Models: (done)
a. Logistic Regression: A simple and interpretable model that can serve as a baseline for the classification task.
b. Multinomial Naive Bayes: Effective for text classification tasks, especially when dealing with text data.
c. Random Forest and XGBoost: Ensemble methods that can handle text data well. They are particularly useful when you have a large feature space after text vectorization. 

2. LSTM:
a. LSTM (Long Short-Term Memory): LSTM networks are suitable for sequences, such as text. They can capture dependencies and context within the text effectively.

3. Transformer Models:
a. BERT (Bidirectional Encoder Representations from Transformers): Pre-trained transformer models like BERT have achieved state-of-the-art results in various NLP tasks, including text classification.

Future work:
1. Try more enseble methods
2. Use pre-trained word embeddings for an nlp model
3. Use a pre-trained transformer model like BioBert of BioGPT, built for medical data use cases.
4. Can filter on specialty more, to improve performance
5. More domain knowledge/business application
"""

# Importing external libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import re
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

pd.set_option('display.max_columns', 500)

# Loading the dataset
data = pd.read_csv('mtsamples.csv', index_col=0)
print(data.shape, 'Original data shape')
print(data['medical_specialty'].value_counts())

####### ----- Data Exploration ----- #######

# Visualize the distribution of target classes
plt.figure(figsize=(10, 6))
data['medical_specialty'].value_counts().plot(kind='bar')
plt.title('Distribution of Specialties')
plt.xlabel('Medical Specialty')
plt.ylabel('Count')
plt.xticks(fontsize=6)  
plt.tight_layout() 
plt.savefig('target_distribution.png')
# This distribution shows that the target label (medical_specialty) is imbalanced. 

plt.figure(figsize=(10, 6))
data_copy = data.copy()
data_copy['transcription'] = data_copy['transcription'].astype(str)
data_copy['text_length'] = data_copy['transcription'].apply(len)
plt.hist(data_copy['text_length'], bins=30)
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Transcription Text Lengths')
plt.tight_layout() 
plt.savefig('transcription_len_distribution.png')

# Check for missing values in the 'specialty' column
missing_specialty = data['medical_specialty'].isna().sum()
print(f"Missing values in 'specialty' column: {missing_specialty}")

# Check for missing values in the 'transcription' column
missing_transcription = data['transcription'].isna().sum()
print(f"Missing values in 'transcription' column: {missing_transcription}")


####### ----- Data Preperation ----- #######
print(data.shape,'1')
data = data.drop(data[data['transcription'].isna()].index)
assert data['transcription'].isna().sum() == 0, "There are missing values in the 'transcription' column."
print(data.shape,'2')

print('******************** Raw data in transcription column: ********************'+'\n')
print('Transcription 1: '+ data['transcription'].iloc[1] +'\n')
print('Transcription 2: ' + data['transcription'].iloc[100] +'\n')
print('Transcription 3: ' + data['transcription'].iloc[1000] +'\n')

def clean_transcription(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char if char not in string.punctuation else ' ' for char in text])
    # Replace brackets with spaces
    text = re.sub(r'[\[\](){}<>]', ' ', text)
    # Replace digits with spaces
    text = ''.join([' ' if char.isdigit() else char for char in text])
    # Remove extra white spaces
    text = ' '.join(text.split())
    return text

def preprocess_text(text):
    # Tokenization: Split the text into words
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization: Reduce words to their base form
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    cleaned_text = ' '.join(words)
    return cleaned_text

data['transcription'] = data['transcription'].apply(clean_transcription)
data['transcription'] = data['transcription'].apply(preprocess_text)
print(data.shape,'3')

print('******************** Transcription column after pre-processing ********************'+'\n')
print('Transcription 1: '+ data['transcription'].iloc[1] +'\n')
print('Transcription 2: ' + data['transcription'].iloc[100] +'\n')
print('Transcription 3: ' + data['transcription'].iloc[1000] +'\n')

####### ----- ML Text Classification Models ----- #######

Xfeatures = data['transcription']
ylabels = data['medical_specialty']

Xfeatures = Xfeatures.reset_index(drop=True)
ylabels = ylabels.to_numpy()


# Define a list of classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression(penalty= 'elasticnet', solver= 'saga', l1_ratio=0.5, random_state=42)),
    ('Multinomial Naive Bayes', MultinomialNB()),
    ('Random Forest', RandomForestClassifier())
    ]

output_dir = 'reports_and_matrices'
os.makedirs(output_dir, exist_ok=True)

n_splits = 3

# Initialize the StratifiedKFold cross-validator
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

for classifier_name, classifier in classifiers:
    fold_accuracies = []
    fold_reports = []
    fold_confusion_matrices = []

    for train_index, test_index in skf.split(Xfeatures, ylabels):
        # Split the data into train and test sets for this fold
        X_train, X_test = Xfeatures.iloc[train_index], Xfeatures.iloc[test_index]
        y_train, y_test = ylabels[train_index], ylabels[test_index]

        print(X_train.shape)
        print(y_train.shape)
        print(X_test.shape)
        print(y_test.shape)

        # Use the same pre-processing and modeling steps as before
        tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)

        # Initialize SMOTE
        smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)

        # Apply SMOTE to the training data of this fold
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, y_train)

        # Create the text classification pipeline
        text_classification_pipeline = Pipeline([('classifier', classifier)])

        # Train the model
        text_classification_pipeline.fit(X_train_resampled, y_train_resampled)

        # Predict on the test set
        y_pred = text_classification_pipeline.predict(X_test_tfidf)

        # Evaluate the model for this fold
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)

        # Append the results for this fold
        fold_accuracies.append(accuracy)
        fold_reports.append(report)
        fold_confusion_matrices.append(confusion)

    # Calculate and print the mean and standard deviation of accuracy across folds
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print(f"Classifier: {classifier_name}")
    print(f"Mean Accuracy: {mean_accuracy}")
    print(f"Standard Deviation of Accuracy: {std_accuracy}")

    # Save the evaluation results for the test set to files
    report_file = os.path.join(output_dir, f"{classifier_name}_test_report.txt")
    with open(report_file, 'w') as file:
        file.write(fold_reports[-1])

    # Save the confusion matrix for the test set to files
    confusion_file = os.path.join(output_dir, f"{classifier_name}_test_confusion.txt")
    with open(confusion_file, 'w') as file:
        file.write(str(fold_confusion_matrices[-1]))


