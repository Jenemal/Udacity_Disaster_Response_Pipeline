# import libraries
import numpy as np
import nltk
import re
import pandas as pd

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle
import sys

# download NLTK data
#nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])

def load_data(database_filepath):
    '''
    PURPOSE: 
    The purpose of this function is to load in the database data and split
    it into X and Y columns for the model. The target_names is used to designate
    parameter names.
    
    INPUT:
    database_filepath - filepath of the database to be loaded in
    
    OUTPUT:
    X - potential input value columns
    Y - output value column
    target_names - key parameter names 
    
    '''

    engine = create_engine(str(database_filepath))  #sqlite:///../data/DisasterResponse.db
    df = pd.read_sql_table('DisasterResponseClean', engine)

    X = df.message
    Y = df.drop(['message', 'original', 'id', 'genre'], axis = 1)
    target_names = Y.columns

    return X, Y, target_names


def tokenize(text):
    '''
    PURPOSE: 
    The purpose of this function is to take the original column text and 
    tokenize it so that we have useful words/phrases to test and train our
    ML model

    INPUT:
    text - text string
    
    OUTPUT:
    final_clean_tokens - array of tokens that will be used in the ML model
    
    '''

    # initialize WordNetLemmatizer   
    lemmatizer = WordNetLemmatizer()

    # replace URLs with blanks
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
   
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with blanks
    for url in detected_urls:
        text = text.replace(url, "")

    # Tokenize text
    tokens = text.split()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = re.sub(r"[^a-zA-Z0-9]", " ", clean_tok)
        clean_tokens.append(clean_tok)  
    
    # Remove stop words
    final_clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return final_clean_tokens


def build_model(grid_search = False):
    '''
    PURPOSE:
    The purpose of this function is to build the pipeline for the ML model.

    INPUT:
    This function does not take an input value
    
    OUTPUT:
    pipeline - machine pipeline for the ML model
    
    '''
    # build the model pipeline
    pipeline = Pipeline([
        ('features',FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect',CountVectorizer(tokenizer = tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    if grid_search == True:
        print('Searching for best parameters...')
        parameters = {
            'clf__estimator__n_estimators': [5, 10, 20]
            , 'clf__estimator__min_samples_split': [2, 3, 4]
        }
        pipeline = GridSearchCV(pipeline, param_grid = parameters)

        # Additional Grid Search parameters that can be used
        # (omitted for faster run time)
        # 'vect__ngram_range': ((1, 1), (1, 2))
        #     , 'vect__max_df': (0.5, 0.75, 1.0)
        #     , 'tfidf__use_idf': (True, False)
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    PURPOSE:
    The purpose of this function is to test the model and evaluate the parameters that
    have the highest weight in determining accurate predictions.

    INPUT:
    model - the model being used (in this case the pipeline)
    X_test - test dataframe input values
    Y_test - test dataframe result value
    category_names - names of the parameters of interest
    
    OUTPUT:
    There are no outputs for this function. However the classification report is printed when the function is called.
    
    '''
    
    Y_pred = model.predict(X_test)
    report  = classification_report(Y_test, Y_pred, target_names=category_names, zero_division=1)
    print(report)


def save_model(model, model_filepath):
    '''
    PURPOSE:
    The purpose of this funciton is to save the model and export it as a pickle file.
    
    INPUT:
    model - model that we want to save
    filepath - filepath where the model will be saved
    
    OUTPUT:
    There is no output for the save_model function
    
    '''
    with open(str(model_filepath), 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(grid_search=True)
        
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