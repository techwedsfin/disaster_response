import sys
import re
import pickle
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine


import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
    This function returns features, labels and label list after loading data from sqllite table to data frame
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("Messages", engine)
    
    # define features and labels for classifier
    X = df['message']
    Y = df.iloc[:,4:]
    
    category_names = list(df.columns[4:])
    
    return X, Y, category_names


def tokenize(text):
    '''
    This function returns reads text as input, convert it to lower case, removes punctuation, 
    splits in words and lemetizes and returns lemetized text
    '''
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    # Split text into words using NLTK
    words = word_tokenize(text)
    
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    
    return lemmed


def build_model():
    '''
    This function builds model based on pipeline and returns model with best parameters
    based on grid searches
    '''
    # build pipeline
    pipeline = Pipeline([('vect',CountVectorizer(tokenizer=tokenize))
                            ,('tfidf',TfidfTransformer())
                            ,('clf',MultiOutputClassifier(RandomForestClassifier(random_state=42)))
                        ])
    #some of the parameters have been commented to make it train faster
    parameters = {
        #'clf__estimator__n_estimators': (100,200),
        #'clf__estimator__criterion': ('gini','entropy'),
        'clf__estimator__min_samples_leaf': (3,5)
        'clf__estimator__random_state': (0,42)
        }


    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function evaluates the model based on f1_score, precision, recall, accuracy
    '''
    # predict on test data
    y_pred = model.predict(X_test)
    
    # derive classification report for model predictions
    for i in range(len(category_names)):
        print("Category : ", category_names[i])
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i]))
        print('Accuracy of {} : {}'.format(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_pred[:,i])))


def save_model(model, model_filepath):
               '''
               This function saves the model to a model file path
               '''
               pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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