import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def load_data(file_path):
    """Load the spam dataset from a CSV file."""
    spam_df = pd.read_csv(file_path, encoding='ISO-8859-1')
    spam_df = spam_df.iloc[:, :2]
    spam_df = spam_df.rename(columns={'v1': 'category', 'v2': 'body'})
    return spam_df

def preprocess_data(df):
    """Split the data into training and testing sets."""
    X = df['body']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return X_train, X_test, y_train, y_test

def create_vectorizer(X_train):
    """Create a TF-IDF vectorizer."""
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    return vectorizer, X_train_tfidf

def train_model(X_train_tfidf, y_train):
    """Create and train a Multinomial Naive Bayes classifier."""
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train_tfidf, y_train)
    return clf

def save_model(clf, vectorizer):
    """Save the trained model and vectorizer."""
    joblib.dump(clf, 'spam_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

def main():
    file_path = 'spam.csv'
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    vectorizer, X_train_tfidf = create_vectorizer(X_train)
    clf = train_model(X_train_tfidf, y_train)
    save_model(clf, vectorizer)
    print("Model and vectorizer saved!")

if __name__ == '__main__':
    main()
