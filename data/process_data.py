import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''Loads messages files and their mapped categories file using the file paths provided, and then merges them'''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")
    return df

def clean_data(df):
    '''Takes a data frame, cleans it by spliting the category column into wide columns and hot encoding values to become ready for MultiClassifier Model'''
    categories = df['categories'].str.split(';', expand=True) # creating a dataframe of the 36 individual category columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames # renaming the columns of `categories`
    for column in categories:

        categories[column] = categories[column].apply(lambda x: x[-1])    # setting each value to be the last character of the string
    
        categories[column] = pd.to_numeric(categories[column]) # convert column from string to numeric
        categories[column] = [1 if x > 0 else 0 for x in categories[column]] #making sure that value is binary, >0 is true, hence setting all possible true values to 1

    df.drop(['categories'], axis=1, inplace=True) #dropping the column where we derived new col names
    df = pd.concat([df,categories], axis=1) # concatenating the original dataframe with the new `categories` dataframe
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    '''Takes a dataframe and database filename to create an sqllite database for persistent storage'''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Message', engine, index=False, if_exists='replace')  


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