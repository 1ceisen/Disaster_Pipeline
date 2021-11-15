import re
import numpy as np
import pandas as pd
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import sys
from sqlalchemy import create_engine


def load(messages_file, categories_file):
    '''
    Input:
        messages_file: Messages data
        categories_file: Categories data
    Output:
        df: Merged dataset from Messages and Categories
    '''
    messages = pd.read_csv(messages_file, encoding = 'utf-8') #Messages
    categories = pd.read_csv(categories_file, encoding = 'utf-8') #Categories
    df = pd.merge(messages, categories, on='id') #Join Categories on Messages
    return df


def clean(df):
    '''
    Input:
        df: Merged dataset from Messages and Categories
    Output:
        df: Cleaned dataset
    '''
    categories = df['categories'].str.split(pat = ';', expand=True) #Split string
    new_header = categories.iloc[0,:] #Obtain headers
    new_header = new_header.str.split(pat = '-') #Clean headers
    categories.columns = [row[0] for row in new_header] #Header names to categories
    for col in categories:
        categories[col] = [row[1] for row in categories[col].str.split(pat = '-')]
        categories[col] = pd.to_numeric(categories[col])
    df = df.drop(['categories'], axis=1) #Drop categories column
    df = pd.concat([df,categories], axis=1) #Concat news categories to df dataframe
    return df


def save(df, database_filename):
    '''
    Save df into sqlite db
    Input:
        df: processed dataset to input
        database_filename: database name
    Output: 
        A SQLite database with table labeled 'DboMessages'
    '''
    eng = create_engine('sqlite:///' + database_filename) #Store sqlite engine call
    df.to_sql('DboMessages', eng, index=False) #Store database object 'DboMessages'

    
def main():
    '''
    Call previously defined functions to produce final dataframe
    '''
    if len(sys.argv) == 4:
        messages_file, categories_file, database_file = sys.argv[1:] #Call database files
        df = load(messages_file, categories_file) #Call load function
        df = clean(df) #Call clean function
        save(df, database_file)    #Call save function     

        
if __name__ == '__main__':
    main()