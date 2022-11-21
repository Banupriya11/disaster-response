import sys
import pandas as pd
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem.wordnet import WordNetLemmatizer
import re
import os
import sqlite3
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    """
    Loads Data from the Database Function
    
    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
    """
    
#     engine = create_engine('sqlite:///' + database_filepath)
#     df = pd.read_sql_table('Message', engine)
    engine = create_engine('sqlite:///Disaster.db')
    df = pd.read_sql_table('Message', engine)  
    X = df ['message']
    y = df.iloc[:,4:]
    return X,y

def tokenize(text):
    """
     Tokenization function. Receives raw text as input which further gets normalized, stop words removed, stemmed and              lemmatized.

    Returns tokenized text
   """        
    text = re.sub( r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    stemmed_words = [PorterStemmer().stem(w) for w in words]
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    lemmed_words = [lemmatizer.lemmatize(word) for word in stemmed_words if word not in stop_words]
    return lemmed_words


# Build a custom transformer which will extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    
    Return a CrossValidation model containing the Pipeline that does text preprocessing
    
    """
    
    pipeline = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer())
        ])),
        ('starting_verb',StartingVerbExtractor())
    ])),
    ('clf',MultiOutputClassifier(AdaBoostClassifier()))
])
     
    parameters = { 'vect__max_df': (0.75,0.85,0.95,1.0),
                    'clf__estimator__n_estimators': [5, 15],
                    'clf__estimator__min_samples_split': [2, 5]
                  }

    cv = GridSearchCV(pipeline, parameters, verbose=True, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test):
    """Print classification report containing precesision, recall and f1
    scores for each outoput column"""
    y_pred = pipeline.predict(X_test)
    for i, col in enumerate(y_test.columns.values):
        
#         print(i,col)      
        print(classification_report(Y_test.loc[:,col], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Saving model with its best parameters using pickle
    """
    
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
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