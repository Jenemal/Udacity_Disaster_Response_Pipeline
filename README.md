# Udacity_Disaster_Response_Pipeline

## Motivation
Using a data set containing real messages that were sent during disaster events I have created a machine learning pipeline to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

## Analysis
This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Files
#### process_data.py is a data cleaning pipeline that:
- Loads the messages and categories datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
#### train_classifier_final.py is a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
#### run.py is the script that:
- runs the disaster response web app
- users can search messages and see what categories they fall under
- users can look at the distribution of message genres and categories
#### messages.csv is a file containing the different disaster messages
#### categories.csv is a file containing the different categories of messages

## To Run
#### process_data.py
Please type the following in the command line. NOTE: filepaths are consistent with the folder structure found in the repository.
python .\process_data.py ..\data\messages.csv  ..\data\categories.csv sqlite:///../data/UdacityProjectNumber2.db
#### train_classifier_final.py
Please type the following in the command line. NOTE: filepaths are consistent with the folder structure found in the repository.
python .\train_classifier_final.py ..\data\categories.csv sqlite:///../data/UdacityProjectNumber2.db classifier_final.pkl
#### run.py
Please type the following in the command line. NOTE: filepaths are consistent with the folder structure found in the repository.
python .\run.py

## Contact Information
Created by Jen E. Malik 02/13/23
