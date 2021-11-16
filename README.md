# Installations
Python 3.8.5

!pip install pandas

!pip install numpy

!pip install re

!pip install nltk

!pip install json

!pip install plotly

!pip install sqlalchemy

!pip install sys

!pip install flask

!pip install pickle

# Business Understanding
Major disasters can occur at anytime and emergency response is critical for ensuring the safety of those impacted. The model developed in this reposity enables a web app to help emergency responders classify text into various categories such that the responses can be classified to the proper personnel. This is done utilizing machine learning models such as Random Forest and Grid Search.

The data is provided from Figure Eight, through Udacity, with two key CSV files provided to help initialize and develop the model:

disaster_categories.csv: A CSV file containing the categories by associated ID

disaster_messages.csv: A CSV file containing the actual messages by associated ID

# Prepare Data

In order to develop an initiated model for production of the web app, a series of steps to clean the data were needed. This included delimiting the disaster_categories csv file to format categories from 1 column to 36 columns, each containing the necessary categories. Addiitonally, a join was conducted to produce a consolidated dataframe between categories and messages. After cleaning this consolidated data frame the data was stored to a sqlite database, upon which it could be called by a train_classifier.py script to develop a machine learning model

# Data Modeling

Data was split into testing and training data whereby a machine learning pipeline was established (using tokenization, TFDIF, mutlioutput classification and a random forest model) to fit the model and predict outcomes on the testing data. Using sklearn's classification_report the f1 score, precision and recall were examined across each category. Some categories in this dataset are imbalanced (eg. water), which could bias the results especially in models performance in predicting true positives. 

# Evaluation of Results

Overall the model has a high degree of accuracy across each category, while the precision and recall vary considerably across each category. Especially where there is the presense of imbalance. Additionally, the model appears to do a proper job in classifying text messages to a proper category.

# How to Interact with this project

## File Description

### data
**disaster_categories.csv**: Categories data in csv file

**disaster_messages.csv**: Messages data in csv file

**DisasterResponse.db**: Processed and merged data frame produced from disaster_categories.csv and disaster_messages.csv files. Stored in sqlite database

**process_data.py**: Python script to process data

### models
**train_classifier.py**: Python script to call DboMessages table in DisasterResponse.db and run machine learning model and store as pickle file

**classifier.pkl**: Pickle file containing machine laearning model

### apps
**run.py**: Script to call data and produce web app

**templates**: Folder containing HTML files for web app

## Instructions

**1.** To run ETL pipeline that cleans data and stores in database 

>python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

**2.** To run ML pipeline that trains classifier and saves

>python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

**3.** Run the following command in the app's directory to run your web app
 
 >python run.py
 
 **4.** Go to http://0.0.0.0:3001/

# Licensing, Authors, Acknowledgements
Credit must be given to Figure Eight for the data. Additional credit must go to Udacity for providing a high level template to construct this project while also providing the necessary teachings. 
