import sys
import pandas as pd
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import numpy as np

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle


def load_data(database_filepath):
    '''Takes a database filepath and loads the data as X,Y, and columns for model training'''
    # loading data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Message',engine)
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    return X,y, y.columns

def tokenize(text):
    '''NLP Tokenizer that takes in text and outputs a tokenized version ready for CountVectorizer and other transformations prior to model training'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''Builds a pipeline with transformations and MultiOutputClassifier, returns a Grid Search CV object that finds best parameters'''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = { #only 1 parameters with 2 variations are used for performance purposes, this can be used to add as many as needed
        'vect__max_df': (0.5,1.0)
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
   
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''Takes in the model and test dataset to make the predictions and evaluate them relative to the truth values'''
    
    Y_pred = model.predict(X_test) #using model to predict based on test data set
    
    i=0
    for cat in category_names: #iterating through each category name where we would need a classification report
        print(cat)
        print(classification_report(Y_test.values[:,i], Y_pred[:,i])) #displaying the classification report for the iterated column
        i = i+1


def save_model(model, model_filepath):
    '''Uses Pickle framework to save the trained model for loading and use later'''
    pickle.dump(model, open(model_filepath, 'wb'))


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