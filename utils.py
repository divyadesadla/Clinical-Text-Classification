"""
File that contains all pre-processing functions
"""
import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import stats
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score


class DataPreprocessor:
    """
    The DataPreprocessor class is designed to prepare and preprocess text data for text classification tasks. It offers the following functionalities:

    1. Data Loading: Reads data from a specified CSV file, considering the 'transcription' and 'medical_specialty' columns.
    2. Data Preparation: Filters out categories with fewer than a specified minimum number of samples and excludes specific categories as defined. This ensures a balanced and relevant dataset for classification.
    3. Text Cleaning: Performs various text cleaning operations, such as removing punctuation, converting to lowercase, and eliminating unnecessary spaces.
    4. Text Preprocessing: Tokenizes text, removes stop words, and lemmatizes words to prepare text data for feature extraction.
    5. Encoding Labels: Encodes the target labels for classification, mapping them to numerical values, and maintaining the label mapping for reference.
    6. Combined Text Cleaning and Preprocessing: Offers a convenient method to apply both cleaning and preprocessing to the text data.
    """

    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, index_col=0)
        self.Xfeatures = self.data['transcription']
        self.ylabels = self.data['medical_specialty']
        self.label_encoder = LabelEncoder()
        self.label_mapping = None

    def prepare_data(self, min_samples=50):   ## You can see the types of medical specialties that fall into the category of notes here / look at the target_distribution.png file to see all specialties, used too investigate those that should be filtered out.
        excluded_categories = [
            ' Consult - History and Phy.',
            ' SOAP / Chart / Progress Notes',
            ' Discharge Summary',
            ' Emergency Room Reports',
            ' Office Notes',
            ' Letters'
        ]
        self.data = self.data.dropna(subset=['transcription'])
        self.data = self.data.groupby('medical_specialty').filter(lambda x: len(x) >= min_samples) # To increase model performance. Can change value or modify further based on need.
        self.data = self.data[~self.data['medical_specialty'].isin(excluded_categories)]
        return self.data

    def clean_text(self, text):
        text = ''.join([char if char not in string.punctuation else ' ' for char in text])
        text = re.sub(r'[\[\](){}<>]', ' ', text)
        text = ''.join([' ' if char.isdigit() else char for char in text])
        text = text.lower()
        text = ' '.join(text.split())
        return text

    def preprocess_text(self, text):
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(words)
        return text

    def clean_and_preprocess_text(self):
        self.data = self.prepare_data()
        self.data['transcription'] = self.data['transcription'].apply(self.clean_text)
        self.data['transcription'] = self.data['transcription'].apply(self.preprocess_text)
        return self.data

    def clean_and_preprocess_text_lstm(self):
        self.data = self.prepare_data()
        self.data['transcription'] = self.data['transcription'].apply(self.clean_text)
        return self.data

    def encode_labels(self, X, y):
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
        encoded_labels = self.label_encoder.transform(y)
        label_mapping = dict(zip(self.label_encoder.transform(self.label_encoder.classes_), self.label_encoder.classes_))
        return encoded_labels, label_mapping



class TradClassifierTrainer:
    def __init__(self, classifiers, X, y, data_preprocessor, model_output, n_splits=2, random_state=42):
        self.classifiers = classifiers
        self.n_splits = n_splits
        self.random_state = random_state
        self.Xfeatures = X
        self.ylabels = y
        self.model_output = model_output
        self.data_processor = data_preprocessor

    def train_and_evaluate(self):
        """
        TradClassifierTrainer is a class for training and evaluating text classification models.

        This class enables the following functionality:
        - Training and evaluating multiple text classification models using cross-validation.
        - Supporting various text classifiers specified as a list of (classifier_name, classifier) tuples.
        - Handling preprocessing of text data, including encoding labels, converting text to TF-IDF features, and applying SMOTE for class imbalance.
        - Generating and saving evaluation metrics, including accuracy, confusion matrices, and classification reports.
        - Producing and displaying mean accuracy and standard deviation across cross-validation folds for each classifier.
        """     
        n_splits = int(self.n_splits)
        results = []

        # Initialize the StratifiedKFold cross-validator
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for classifier_name, classifier in self.classifiers:
            fold_accuracies = []
            fold_precision = []
            fold_recall = []
            fold_f1 = []
            fold_specificity = []
            fold_mcc = []
            fold_kappa = []

            for train_index, test_index in skf.split(self.Xfeatures, self.ylabels):

                # Split the data into train and test sets for this fold
                X_train, X_test = self.Xfeatures.iloc[train_index], self.Xfeatures.iloc[test_index]
                y_train, y_test = self.ylabels[train_index], self.ylabels[test_index]

                # Call DataPreprocessor to encode labels within the cross-validation loop
                encoded_labels_train, label_mapping_train = self.data_processor.encode_labels(X_train, y_train)
                encoded_labels_test, label_mapping_test = self.data_processor.encode_labels(X_test, y_test)

                # Converting text data into a numerical format (TF-IDF features)
                tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
                X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
                X_test_tfidf = tfidf_vectorizer.transform(X_test)

                # Initialize SMOTE
                smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=2)

                # Apply SMOTE to the training data of this fold
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train_tfidf, encoded_labels_train)

                # Create the text classification pipeline
                text_classification_pipeline = Pipeline([('classifier', classifier)])

                # Train the model
                text_classification_pipeline.fit(X_train_resampled, y_train_resampled)

                # Predict on the test set
                y_pred = text_classification_pipeline.predict(X_test_tfidf)

                # Evaluate the model for this fold
                accuracy = accuracy_score(encoded_labels_test, y_pred)                
                precision = precision_score(encoded_labels_test, y_pred, average='weighted')
                recall = recall_score(encoded_labels_test, y_pred, average='weighted')
                f1 = f1_score(encoded_labels_test, y_pred, average='weighted')
                confusion = confusion_matrix(encoded_labels_test, y_pred)
                TN = confusion[0, 0]  # True Negatives
                FP = confusion[0, 1]  # False Positives
                specificity = TN / (TN + FP)
                mcc = matthews_corrcoef(encoded_labels_test, y_pred)
                kappa = cohen_kappa_score(encoded_labels_test, y_pred)

                self.model_output.save_confusion_matrix(encoded_labels_test, y_pred, label_mapping_test, f"{classifier_name}_confusion_matrix.png")
                self.model_output.save_classification_report(encoded_labels_test, y_pred, label_mapping_test, f"{classifier_name}_classification_report.txt")

                # Append the results for this fold
                fold_accuracies.append(accuracy)
                fold_precision.append(precision)
                fold_recall.append(recall)
                fold_f1.append(f1)
                fold_specificity.append(specificity)
                fold_mcc.append(mcc)
                fold_kappa.append(kappa)


            # Calculate and print the mean and standard deviation of accuracy across folds
            mean_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            mean_precision = np.mean(fold_precision)
            mean_recall = np.mean(fold_recall)
            mean_f1 = np.mean(fold_f1)
            mean_specificity = np.mean(fold_specificity)
            mean_mcc = np.mean(fold_mcc)
            mean_kappa = np.mean(fold_kappa)
            print(f"Classifier: {classifier_name}")
            print(f"Mean Accuracy: {mean_accuracy}")
            print(f"Standard Deviation of Accuracy: {std_accuracy}")
           
            # Storing results
            result = {
                'Classifier': classifier_name,
                'Mean Accuracy': mean_accuracy,
                'Std Deviation of Accuracy': std_accuracy,
                'Mean Precision': mean_precision,
                'Mean Recall': mean_recall,
                'Mean F1-Score': mean_f1,
                'Mean Specificity': mean_specificity,
                'Mean MCC': mean_mcc,
                'Mean Kappa': mean_kappa
                }
            results.append(result)
        return results


class ModelOutput:
    def __init__(self):
        pass

    @staticmethod
    def save_confusion_matrix(y_true, y_pred, label_mapping, filename):
        if not label_mapping:
            print("No label mapping provided.")
            return

        if not set(y_true).issubset(set(label_mapping.keys())) or not set(y_pred).issubset(set(label_mapping.keys())):
            print("Labels in y_true or y_pred are not found in the label mapping.")
            return

        labels = [label_mapping[label] for label in y_true]
        predicted_labels = [label_mapping[label] for label in y_pred]
        cm = confusion_matrix(labels, predicted_labels)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        sns.heatmap(cm, annot=True, cmap="Greens", ax=ax, fmt='g')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        ax.xaxis.set_ticklabels(label_mapping.values())
        ax.yaxis.set_ticklabels(label_mapping.values())
        plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def save_classification_report(y_true, y_pred, label_mapping, filename):
        if not label_mapping:
            print("No label mapping provided.")
            return

        if not set(y_true).issubset(set(label_mapping.keys())) or not set(y_pred).issubset(set(label_mapping.keys())):
            print("Labels in y_true or y_pred are not found in the label mapping.")
            return

        y_true_labels = [label_mapping[label] for label in y_true]
        y_pred_labels = [label_mapping[label] for label in y_pred]
        report = classification_report(y_true_labels, y_pred_labels, target_names=label_mapping.values(), zero_division=1)
        with open(filename, 'w') as file:
            file.write(report)












