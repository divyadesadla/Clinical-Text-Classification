Medical Transcription Classification:
This project focuses on classifying medical transcriptions into various subspecialties of medicine. The goal is to create a text classification model that can accurately determine the subspecialty for each medical transcription.

Dataset:
The dataset used for this project is the Transcription Samples (MTSamples). It contains de-identified medical transcriptions, each labeled with a subspecialty of medicine. Some transcriptions are labeled as notes, which are not classified into specific subspecialties. The objective is to classify the transcriptions into relevant medical subspecialties.

Project Structure:
The project consists of two main Python files:

models.py: This file contains the code for training and evaluating text classification models. It uses various classifiers, including Logistic Regression, Multinomial Naive Bayes, and Random Forest, to predict the subspecialty labels for the transcriptions. It also handles data preprocessing, encoding labels, and generating evaluation metrics.

utils.py: This file contains utility functions for data preprocessing and model evaluation. It provides data cleaning and preprocessing methods, label encoding, and functions for saving confusion matrices and classification reports.

explore.py: This file contains some of the exploratory work I did before setting up the main pipeline (i.e. models.py and utils.py). You dont need to run this code.

Getting Started:
Before running the project, ensure that you have the following prerequisites:
1. Python 3.8+
2. Required Python libraries mentioned in the models.py and utils.py files. You can use the requirements.txt file to setup your ennviroment.
3. Download the MTSamples dataset save as 'mtsamples.csv' in the project directory.

Installation:
1. Clone the project repository:
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
2. Install the required Python libraries by running the following command:
pip install -r requirements.txt

Usage:
1. To train and evaluate the text classification models, follow these steps:
Run the models.py script, which trains and evaluates the classifiers using cross-validation:
python3 models.py
2. The script will provide results for each classifier, including mean accuracy, standard deviation of accuracy, mean precision, mean recall, mean F1-score, mean specificity, mean MCC, and mean Kappa.

Contact
If you have any questions, feedback, or need further assistance, feel free to contact me at [ddesadla@gmail.com].