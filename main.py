import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
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
    processed_texts = [" ".join([stemmer.stem(word) for word in text.split() if word.lower() not in stop_words]) for text in texts]
    return processed_texts


# Train and evaluate multiple models
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model'):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'{model_name} Evaluation')
    print(classification_report(y_test, predictions))


if __name__ == '__main__':

    # Read data from file
    df = pd.read_csv('./offenseval-training-v1.tsv', sep='\t')
    # Data extraction from data file
    texts = df['tweet'].values
    labels = df['subtask_a'].values

    # Preprocessing of texts
    texts_processed = preprocess(texts)

    # Vectorization of texts
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts_processed)
    y = labels

    # Split data into training and testing sets (70% training, 30% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)




    # Naive Bayes
    train_and_evaluate_model(MultinomialNB(), X_train, y_train, X_test, y_test, 'Naive Bayes')

    # Decision Tree
    train_and_evaluate_model(DecisionTreeClassifier(), X_train, y_train, X_test, y_test, 'Decision Tree')

    # Random Forest
    train_and_evaluate_model(RandomForestClassifier(), X_train, y_train, X_test, y_test, 'Random Forest')

    # SVM
    train_and_evaluate_model(SVC(), X_train, y_train, X_test, y_test, 'SVM')

    # Multilayer Perceptron
    train_and_evaluate_model(MLPClassifier(max_iter=1000), X_train, y_train, X_test, y_test, 'Multilayer Perceptron')


