import os
import pandas as pd
import numpy as np
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from utils import DataPreprocessor, TradClassifierTrainer, ModelOutput

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Define a list of classifiers
classifiers = [
    ('Logistic Regression', LogisticRegression(solver='lbfgs', max_iter=500, C=1.0, class_weight='balanced')),
    ('Multinomial Naive Bayes',MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)),   
    ('Random Forest', RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_leaf=1, class_weight=None))      
]


def train():
    # Loading the dataset and pre-processing the data
    data_file = 'mtsamples.csv'
    data_preprocessor = DataPreprocessor(data_file)
    model_data = data_preprocessor.clean_and_preprocess_text()
    model_data = model_data.reset_index(drop=True)
    X = model_data['transcription']
    y = model_data['medical_specialty']

    # Initialize a list to store results
    results = []
    model_output = ModelOutput()

    # Create an instance of TradClassifierTrainer and run training and evaluation
    classifier_trainer = TradClassifierTrainer(classifiers, X, y, data_preprocessor, model_output)
    results += classifier_trainer.train_and_evaluate()

    # Save the results to a text file in the current folder
    results_filename = 'classification_results.txt'
    with open(results_filename, 'w') as file:
        for result in results:
            file.write(f"Classifier: {result['Classifier']}\n")
            file.write(f"Mean Accuracy: {result['Mean Accuracy']}\n")
            file.write(f"Standard Deviation of Accuracy: {result['Std Deviation of Accuracy']}\n")
            file.write(f"Mean Precision: {result['Mean Precision']}\n")
            file.write(f"Mean Recall: {result['Mean Recall']}\n")
            file.write(f"Mean F1-Score: {result['Mean F1-Score']}\n")
            file.write(f"Mean Specificity: {result['Mean Specificity']}\n")
            file.write(f"Mean MCC: {result['Mean MCC']}\n")
            file.write(f"Mean Kappa: {result['Mean Kappa']}\n\n")

if __name__ == "__main__":
    train()
