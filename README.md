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

### 1. Loading and Preprocessing

- **Loading Data**: The notebook starts by loading the dataset from the provided source. Ensure that you have the dataset file placed correctly before running the code.

- **Preprocessing Steps**:
  - **Text Cleaning**: Remove any unwanted characters and normalize text.
  - **Tokenization**: Break text into individual words or tokens.
  - **Stopword Removal**: Remove common words that do not contribute to the meaning of the text.

  These preprocessing steps are crucial as they help in transforming raw text into a clean format suitable for feature extraction and model training. Follow the markdown explanations in the notebook to understand the specific techniques and their impact on model performance.

### 2. Feature Extraction

- **TfidfVectorizer**: The notebook demonstrates how to use the `TfidfVectorizer` from the `scikit-learn` library to convert text data into numerical features.
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**: This method transforms text into a matrix of features where each term is weighted based on its importance in the document relative to the corpus.

  The feature extraction step is essential for converting text data into a format that machine learning models can work with. Refer to the notebook for detailed code and explanations on how this transformation is performed.

### 3. Model Development

- **Naive Bayes and SVM Models**:
  - **Naive Bayes**: A probabilistic model that works well with text classification by assuming feature independence. The notebook includes code for training and evaluating this model.
  - **Support Vector Machine (SVM)**: A robust classifier that finds the optimal hyperplane to separate different classes. The notebook demonstrates how to train and evaluate an SVM model.

- **Model Evaluation**:
  - **Metrics**: Accuracy, F1-score, and other relevant metrics are calculated to evaluate model performance.
  - **Comparison**: Results from both models are compared to determine the best performing model for the emotion classification task.
 
### 4. Predicting New Sentences:

Use the provided function to predict the emotion of new text samples.

  This section of the notebook provides a comprehensive guide to training and evaluating the machine learning models. Follow the markdown explanations for insights into the model evaluation process and interpretation of results.

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

