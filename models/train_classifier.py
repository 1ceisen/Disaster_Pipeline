import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sklearn.metrics import confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import sys
from sqlalchemy import create_engine
import pickle


def load(database_filepath):
    '''
    Load data from generated database as dataframe
    Input:
        database.csv: File path of sql database
    Output:
        X: Messages
        Y: Categories
        categories: Unique categories
    '''
    eng = create_engine('sqlite:///' + database_filepath) #Database engine
    df = pd.read_sql_table('DboMessages', eng) #Read DisasterMessages table
    X = df['message'] #Messages
    Y = df[df.columns[4:]] #Categories
    categories = list(df.columns[4:]) #Category names
    return X, Y, categories


def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: Message text
    Output:
        clean_tokens: Cleaned, tolenized and lemmatized text
    '''
    print("tokenize...") #print while processing
    tokens = word_tokenize(text) #tokenize text
    lem = WordNetLemmatizer() #initialize lemmatizer   
    clean_tokens = [] #initialize empty list for clean tokens needed for loop
    for tok in tokens:
        clean_tok = lem.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build(grid_search_cv = False):
    '''
    Build the model with pipeline and gridsearch for parameter optimization
    Args:
        grid_search_cv (bool): if True will conduct grid search for best parameters
    Returns:
        pipeline: model after pipeline and gridsearch
    '''
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)), 
                     ('tfidf', TfidfTransformer()), 
                     ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ]) #initialize functions in pipeline
    if grid_search_cv == True:
        print('best parameters...')
        param = {'vect__ngram_range': ((1, 1), (1, 2))
            , 'vect__max_df': (0.5, 0.75, 1.0)
            , 'tfidf__use_idf': (True, False)
            , 'clf__estimator__min_samples_split': [2, 3, 4]
        } # grid search parameters from pipeline.get_params().keys()
        pipeline = GridSearchCV(pipeline, param_grid = param) #Store gridsearch to pipeline
    return pipeline


def evaluate(model, X_test, Y_test, categories):
    '''
    Evaluate the model performances and print the results
    Args:
        model: model
        X_test: Messages dataset
        Y_test: Categories dataset
        categories: Name of categories
    '''
    Y_pred = model.predict(X_test) #Predict on test data
    # Calculate the accuracy for each of them.
    length = len(categories)
    for i in range(length):
       print('Category: {} '.format(categories[i]))
       print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
       print('Accuracy = {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i]))) #Loop to return results


def save(model, model_file):
    '''
    Save model to a pickle file
    Args:
        model: model
        model_pickle_filename: Pickle file
    '''
    pickle.dump(model, open(model_file, 'wb')) #dump model to pickle file


def load_model(model_pickle_file):
    '''
    Return model from pickle file
    Args:
        model_pickle_file: Pickle file
    Returns:
        model: Pickle file model
    '''
    return pickle.load(open(model_pickle_file, 'rb')) #call pickle model


def main():
    '''
    Call previously defined functions to produce final dataframe
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:] #Call database and model file
        X, Y, categories = load(database_filepath)  #Call load function
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #Split train and test
        model = build() #Call build function
        model.fit(X_train, Y_train) #Fit model
        evaluate(model, X_test, Y_test, categories) #Call evaluate function
        save(model, model_filepath) #Call save function


if __name__ == '__main__':
    main()