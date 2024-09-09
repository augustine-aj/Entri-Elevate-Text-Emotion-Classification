# Entri-Elevate-Text-Emotion-Classification

# Emotion Classification in Text

## Objective
The objective of this project is to build machine learning models to classify emotions in text samples. The project involves data preprocessing, feature extraction, model training, and evaluation using the provided dataset. The goal is to compare different models and predict emotions in new text samples.

## Dataset
The dataset for this project can be downloaded from this link. It is a CSV file containing text samples and their corresponding emotion labels.

https://drive.google.com/file/d/1HWczIICsMpaL8EJyu48ZvRFcXx3_pcnb/view

## Key Components
- Loading and Preprocessing:

Load the dataset and perform text cleaning, tokenization, and removal of stopwords.
Preprocessing helps in normalizing the text data, which improves model performance.
- Feature Extraction:

Use TfidfVectorizer to convert text data into numerical features.
TF-IDF transforms text into a format that can be used by machine learning algorithms.
- Model Development:

Train and evaluate two machine learning models:
Naive Bayes: A probabilistic classifier based on Bayes' theorem.
Support Vector Machine (SVM): A model that finds the best hyperplane to separate classes.
- Model Comparison:

Evaluate the models using metrics like accuracy and F1-score.
Compare the performance of Naive Bayes and SVM models.
Predicting New Sentences:

Use the trained models to classify emotions in new text samples.

## How to Use
1. Loading and Preprocessing:

The notebook includes code to load and preprocess the dataset.
Follow the markdown explanations in the notebook for details on preprocessing techniques.
2. Feature Extraction:

The notebook demonstrates how to use TfidfVectorizer for feature extraction.
3. Model Development:

The notebook contains sections for training Naive Bayes and SVM models.
Evaluation metrics are calculated and compared.
4. Predicting New Sentences:

Use the provided function to predict the emotion of new text samples.

## Results
The results section of the notebook will provide:

* Performance metrics (accuracy and F1-score) for each model.
* Comparison between Naive Bayes and SVM models.
* Predictions for new text samples.

## Conclusion
In this project, we explored emotion classification in text using two machine learning models: Naive Bayes and Support Vector Machine (SVM). We preprocessed the dataset by cleaning and tokenizing the text, and used TF-IDF vectorization to convert text into numerical features. Both models were trained and evaluated, and their performance was compared based on accuracy and F1-score.

* Naive Bayes: Known for its simplicity and efficiency, Naive Bayes provided a solid baseline for text classification tasks.
* SVM: Leveraged its capability to handle high-dimensional data effectively, showing competitive performance in classifying emotions.
The ability to classify emotions from text has practical applications in various domains, including customer feedback analysis, social media monitoring, and mental health assessment. Future improvements could include exploring additional models, fine-tuning hyperparameters, and expanding the dataset for better generalization.

