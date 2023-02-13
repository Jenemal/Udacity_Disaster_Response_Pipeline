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
#### train_classifier.py is a machine learning pipeline that:
- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file

## Contact Information
Created by Jen E. Malik 02/13/23
