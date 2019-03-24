import sys
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sqlalchemy   import create_engine

def load_data(database_filepath):
    """
    Load the SQLite database from the database_filepath. Separate it to two parts:
    (1) message input and (2) message type labels.

    INPUTS:
        database_filename: path of SQLite database of cleaned messeage table

    RETURN:
        X: inputs of messages
        Y: labels of message categories
        column_name: the categories names
    """

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('DisasterResponse',con=engine)
    X = df.message
    Y = df.iloc[:,4:]
    column_name = list(Y.columns.tolist())

    return X, Y, column_name


def tokenize(text):
    """
    - transform the sentence into word tokens
    - lemmatized the tokens

    INPUTS:
        text: message text to be tokened

    RETURN:
        clean_tokens: the list of cleaned and tokenized words
    """


    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():

    """
    Builds the pipeline to tag the meesages into different categories. GridSearchCV
    is also applied to fin-tuned the model.


    RETURN:
        cv: model after fine-tuned after GridSearchCV
    """

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])

    parameters =  {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.75, 1.0),
        #'tfidf__use_idf': (True, False),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)



    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    """
    Evaluate the model performance on the test data. Print the results on precision,
    recall, and f1 score for each category.

    INPUTS:
        model: fine-tuned classification model after GridSearchCV
        X_test: test sample of the messeages
        Y_test: labels of the messeages form the test sample
        category_names: messeage categories' names


    """

    predict_cv = model.predict(X_test)
    Y_pred_cv = pd.DataFrame(predict_cv, columns = category_names)

    for i in range(36):
        print(category_names[i],\
          '\n',\
          classification_report(Y_test.iloc[:,i], Y_pred_cv.iloc[:,i]))



def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))




def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

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
