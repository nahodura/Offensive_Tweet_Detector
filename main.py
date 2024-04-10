import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

"""
Pour ce TP, le but est de construire des modèles pour détecter si un tweet sur Twitter contient du langage offensif.
    @ entrée : texte brut d'un tweet
    @ sortie : OFF si le tweet est offensif, NOT sinon

Tester les algorithmes présentés dans le cours et quelques variantes
- Naive Bayes
- Arbre de décision
- Forêt aléatoire
- SVM
- Perceptron multicouche,
- Modèle basé sur BERT (BONUS)
"""


# 1. Preprocessing function : Tokenization, stemming, and stopwords removal
def preprocess(texts):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    processed_texts = [" ".join([stemmer.stem(word) for word in text.split()
                                 if word.lower() not in stop_words]) for text in texts]
    return processed_texts


# 2. Function to train and evaluate a model with different hyperparameters
def train_and_evaluate_model(model, params, X_train, y_train, X_test, y_test, model_name='Model'):
    # Grid search for hyperparameter tuning : n_jobs=-1 for parallel processing and verbose=3 for detailed output.
    grid_search = GridSearchCV(model, params, cv=5, scoring='f1_macro', n_jobs=-1, verbose=10)
    # Train the model on the training set
    grid_search.fit(X_train, y_train)
    print(f'{model_name} Best Params:', grid_search.best_params_)
    # Evaluate the model on the test set
    predictions = grid_search.predict(X_test)

    # Print classification report
    print(f'{model_name} Evaluation')
    print(classification_report(y_test, predictions))
    return grid_search.best_estimator_


if __name__ == '__main__':
    # ----------------------------- Data loading and preprocessing -----------------------------
    df = pd.read_csv('./offenseval-training-v1.tsv', sep='\t')
    # Data extraction from data file
    texts = df['tweet'].values
    labels = df['subtask_a'].values

    # Preprocessing of texts
    texts_processed = preprocess(texts)

    # ----------------------------- Vectorization of texts -----------------------------------
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts_processed)
    y = labels

    # --------------------Split data into training and testing sets ------------------------
    TrainingDataSet, TestDataSet, TrainingLabels, TestLabels \
        = train_test_split(X, y, test_size=0.3, random_state=42)

    # ----------------------------- Parameters for Grid Search -----------------------------
    nb_params = {
        'alpha': [0.5, 1.0, 2.0, 5.0, 10.0],  # Multinomial Naive Bayes
    }
    dt_params = {
        'max_depth': [None, 10, 20, 30, 40, 50],  # None for no limit on the depth
        'criterion': ['gini', 'entropy'],  # 'gini' for small datasets and 'entropy' for large datasets
    }
    rf_params = {
        'n_estimators': [10, 50, 100],  # Number of trees in the forest
        'max_depth': [None, 10, 20],
        'criterion': ['gini', 'entropy'],
    }
    svm_params = {
        'kernel': ['linear', 'rbf'],  # 'linear' for small datasets and 'rbf' for large datasets
        'C': [0.1, 1, 10, 100]
    }
    mlp_params = {
        'hidden_layer_sizes': [(50,), (100,)],  # 50 and 100 neurons in the hidden layer
        'activation': ['logistic', 'relu'],  # 'logistic' for small datasets and 'relu' for large datasets
        'solver': ['adam', 'sgd', 'lbfgs']
    }

    # ---------------------- Training and eval. of models -------------------------
    """
    # Naive Bayes
    best_nb = train_and_evaluate_model(MultinomialNB(), nb_params, TrainingDataSet, TrainingLabels, TestDataSet,
                                       TestLabels, 'Naive Bayes')
    
    # On doit convertir les données d'entraînement et de test en matrices denses pour le Naive Bayes gaussien
    TrainingDataSetDense = TrainingDataSet.toarray()
    TestDataSetDense = TestDataSet.toarray()
    
    # Gaussian Naive Bayes
    best_gnb = train_and_evaluate_model( GaussianNB(),                   
        {},  # Pas de paramètres à régler pour le Naive Bayes gaussien
        TrainingDataSetDense, TrainingLabels, TestDataSetDense,TestLabels, 'Gaussian Naive Bayes')
    
    # Decision Tree
    best_dt = train_and_evaluate_model(DecisionTreeClassifier(), dt_params, TrainingDataSet, TrainingLabels,
                                       TestDataSet, TestLabels, 'Decision Tree')
    # Random Forest
    best_rf = train_and_evaluate_model(RandomForestClassifier(), rf_params, TrainingDataSet, TrainingLabels,
                                       TestDataSet, TestLabels, 'Random Forest')
                                       """
    # SVM
    best_svm = train_and_evaluate_model(SVC(), svm_params, TrainingDataSet, TrainingLabels, TestDataSet, TestLabels,
                                        'SVM')

    # Multilayer Perceptron
    best_mlp = train_and_evaluate_model(MLPClassifier(max_iter=200), mlp_params, TrainingDataSet, TrainingLabels,
                                        TestDataSet, TestLabels, 'Multilayer Perceptron')

    # Multilayer Perceptron avec early stopping
    best_mlp = train_and_evaluate_model(
        MLPClassifier(max_iter=300, early_stopping=True, hidden_layer_sizes=(50,)),
        mlp_params,
        TrainingDataSet,
        TrainingLabels,
        TestDataSet,
        TestLabels,
        'Multilayer Perceptron'
    )


