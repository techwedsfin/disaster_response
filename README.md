# Disaster Response Pipeline Project

###Description:- This project builds a machine learning pipeline to categorize emergency messages based on needs communicated by the sender

Installations:-This code runs with Python version 3.x and requires below listed packages 
Flask==0.12.4
nltk==3.2.5
numpy==1.12.1
pandas==0.23.3
plotly==2.0.15
requests==2.21.0
scikit-learn==0.19.1
SQLAlchemy==1.2.18

###Project Motivation:-This is an Udacity Nanodegree project. In this project i applied some of the skills like software engineering, data engineering, natural language processing, and machine learning skills that i learn to analyze message data during disasters and build a model for an API that classifies disaster messages.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
