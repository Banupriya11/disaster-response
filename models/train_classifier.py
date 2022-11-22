import sys
import os
import re
import pickle
from sqlalchemy import create_engine
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

nltk.download(['punkt', 'wordnet', 'stopwords'], quiet=True)

def load_data(database_filepath):
    """
    Load data from SQLite file and returns X, Y as DataFrames 
    
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Message', engine)
    
#     engine = create_engine('sqlite:///' + database_filepath)
#     table_name = os.path.basename(database_filepath).replace(".db","") + "_table"
#     df = pd.read_sql_table(table_name,engine)
#     engine = create_engine('sqlite:///Disaster.db')
#     df = pd.read_sql_table('Message', engine)
    X = df ['message']
    Y = df.iloc[:,4:]
    Y = Y.astype(bool)
    return X, Y



def tokenize(text):
    """
    Tokenization function. Receives raw text as input which further gets normalized, then stop words get removed, stemmed and lemmatized.
    Returns tokenized text.
    
    """
    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    stemmed_words = [PorterStemmer().stem(w) for w in words]
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    lemmed_words = [lemmatizer.lemmatize(word) for word in stemmed_words if word not in stop_words]
    return lemmed_words


def build_model():
    """
    Return a CrossValidation model containing the Pipeline that does text preprocessing
    """
    
    pipeline = Pipeline([
        ('count', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier(n_estimators=15, max_depth=4, n_jobs=-1))
    ])
    parameters = {
        'count__ngram_range': [(1,1), (1, 3)],
        'clf__n_estimators': [5, 8],
        'clf__max_depth': [2, 5]
    }
    cv = GridSearchCV(pipeline, parameters, verbose=True, cv=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Print classification report which includes precesision, recall and f1
    scores for each outoput column
    """
    
    Y_pred = model.predict(X_test)
    for i, col in enumerate(Y_test.columns.values):
        
        print('---{}---'.format(col.upper()))
        print(classification_report(Y_test.loc[:,col], Y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the model object as a pickle file
    """
    
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    """
    Read the command line arguments and execute model training steps
    """
    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()