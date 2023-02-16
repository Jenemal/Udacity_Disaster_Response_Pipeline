# import libraries
import pandas as pd
from sqlalchemy import create_engine

import sys


def load_data(messages_filepath, categories_filepath):
    '''
    PURPOSE: 
    The purpose of the function is to load in the two datasets and merge
    them into one dataframe.
    
    INPUT:
    messages_filepath - text string with the filepath for the messages.csv file
    categories_filepath - text string with the filepath for the categories.csv file
    
    OUTPUT:
    df - dataframe containing the columns from the categories and messages csv files
    '''
    # read in the two csv files
    messages = pd.read_csv(str(messages_filepath), low_memory = False)
    categories = pd.read_csv(str(categories_filepath), low_memory = False)

    # merge the two dataframes on id
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    '''
    PURPOSE: 
    The purpose of the function is to clean the dataframe

    INPUT:
    df - raw dataframe
    
    OUTPUT:
    df_cleaned - cleaned dataframe
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]
    tmp = row.tolist()

    # extract a list of new column names for categories
    category_colnames = [x[:-2] for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        tmp = categories[column].astype(str)
        tmp = tmp.str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(tmp)
    
    # drop the original categories column from `df`
    df_cleaned = df.drop(columns=['categories'])

    # concatenate the original dataframe with the new `categories` dataframe
    df_cleaned = pd.concat([df_cleaned,categories], axis=1)

    # check for and remove duplicates
    while df_cleaned.duplicated().sum() > 0:
        # drop duplicates
        bool_series = df_cleaned.duplicated(keep='first')
        df_cleaned = df_cleaned[~bool_series]

    # replace values of 2 so df only has 1s and 0s
    df_cleaned.replace(2,1, inplace=True)

    return df_cleaned
    


def save_data(df, database_filename):
    '''
    PURPOSE: 
    The purpose of the function is to create the database engine and then save the database

    INPUT:
    df - dataframe to save
    database_filename - the filepath of the database to save the cleaned data
    
    OUTPUT:
    There is no model output
    '''
    
    # create the database engine
    engine = create_engine(str(database_filename)) # sqlite:///../data/DisasterResponse.db
    # save the database
    df.to_sql('DisasterResponseClean', engine, index=False, if_exists="replace")


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()