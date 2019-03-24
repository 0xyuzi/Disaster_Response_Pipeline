# Disaster_Response_Pipeline
## Project Description

In this project, you'll apply these skills to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.
A data set containing real messages that were sent during disaster events. A machine learning pipeline to categorize these events so that messages to an appropriate disaster relief agency.
The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data

## File Description

```
Disaster_Response_Pipeline
|- app
  | - template
     |- master.html  # main page of web app
     |- go.html  # classification result page of web app
  |- run.py  # Flask file that runs app

|- data
  |- disaster_categories.csv  # data to process 
  |- disaster_messages.csv  # data to process
  |- process_data.py
  |- ETL Pipeline Preparation.ipynb
  |- DisasterResponse.db   # database to save clean data to

|- models
  |- train_classifier.py
  |- classifier.pkl  # saved model
  |- ML Pipeline Preparation.ipynb  # saved model

```

## Instructions 

1. Under the _data_ directory, run the ETL pipeline that cleans and store the cleaned data in to SQLite database in the command prompt  ```python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db ```
2. Under the _model_ directory, run the ML pipeline to train the XGBClassifier with GridSearchCV and save to the pickle file in the command prompt  ``` python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl ```
3. Under the the _app_ directory, run the web app in the command prompt ``` python run.py```
4. Open the [http://localhost:3001](http://localhost:3001) to check the web app.

## Installations

NLTK,SQLAlchecmy, sklearn, xgboost

## Licensing
It is available below under MIT license.

## Acknowledge
This project is under the Udacity Data Science Nanodegree.
