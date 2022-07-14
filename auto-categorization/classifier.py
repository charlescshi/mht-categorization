import re
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

import PyPDF2 as pdf
import csv
import os
import pickle
import tqdm

RS = 8008135

class ClfSwitcher(BaseEstimator):
    
    def __init__(self, estimator=RandomForestClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        
        Parameters
        ----------
        estimator: sklearn object, the classifier
        """
        self.estimator = estimator
    
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def score(self, X, y):
        return self.estimator.score(X, y)

def clean_text(s):
    s = re.sub(r'\n\-\n', '', s)
    s = re.sub(r'[^\w\-]+', ' ', s)
    s = s.replace(u'\u0160', ' ')
    return s

def get_clean_text(fname):
    doc = pdf.PdfFileReader(fname)
    full_text = ''
    for page_num in range(doc.getNumPages()):
        page = doc.getPage(page_num)
        page_text = page.extractText()
        page_text = clean_text(page_text)
        full_text += page_text + ' '
    return full_text

def read_data(fname):
    array = []
    with open(fname, encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue

            id_ = row[0]
            categories = row[6]
            if categories:
                categories = categories.split(';')
                categories = [s.replace(' ', '') for s in categories]
                structural = lab = fe = rdd = diff = iv = event = na = 0
                if 'structural' in categories:
                    structural = 1
                if 'lab' in categories:
                    lab = 1
                if 'afe' in categories:
                    fe = 1
                if 'ffe' in categories:
                    fe = 1
                if 'nfe' in categories:
                    fe = 1
                if 'rdd' in categories:
                    rdd = 1
                if 'diff-in-diff' in categories:
                    diff = 1
                if 'iv' in categories:
                    iv = 1
                if 'event-studies' in categories:
                    event = 1
                if 'na' in categories:
                    na = 1
                l = [id_, structural, lab, fe, rdd, diff, iv, event, na]
                if 1 in l:
                    array.append(l)
    df = pd.DataFrame(array, columns=['id', 'structural', 'lab', 'fe', 'rdd', 'diff-in-diff', 'iv', 'event-studies', 'na'])
    return df

def make_full_df():
    classified = read_data('cat.csv')
    array = []
    num_files = len(os.listdir('papers/'))
    i = 0
    for fname in os.listdir('papers/'):
        print('[%i/%i]' % (i, num_files))
        i += 1

        id_ = re.findall(r'[0-9]+', fname)[0]
        try:
            text = get_clean_text('papers/' + fname)
            array.append([id_, text])
        except Exception as e:
            print('uh oh: ' + str(e))
    text_df = pd.DataFrame(array, columns=['id', 'text'])
    df = pd.merge(text_df, classified, on='id')
    with open('full_df.pkl', 'wb') as f:
        pickle.dump(df, f)
    
def score(y_true, y_pred, index):
    """Calculate precision, recall, and f1 score"""
    
    metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    performance = {'precision': metrics[0], 'recall': metrics[1], 'f1': metrics[2]}
    return pd.DataFrame(performance, index=[index])

def test_models():
    with open('full_df.pkl', 'rb') as f:
        df = pickle.load(f)
    tags = df[['structural', 'lab', 'fe', 'rdd', 'diff-in-diff', 'iv', 'event-studies', 'na']]
    X_train, X_test, y_train, y_test = train_test_split(df['text'], tags, train_size=0.8, stratify=None, random_state=RS)

    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ClfSwitcher())])

    grid = ParameterGrid({
    'clf__estimator': [
        MultiOutputClassifier(LogisticRegression(class_weight='balanced', random_state=RS), n_jobs=-1),
        MultiOutputClassifier(SGDClassifier(class_weight='balanced', random_state=RS, loss='modified_huber'), n_jobs=-1),
        MultiOutputClassifier(LinearSVC(class_weight='balanced', random_state=RS), n_jobs=-1),
        KNeighborsClassifier(n_jobs=-1),
        RandomForestClassifier(class_weight='balanced', random_state=RS, n_jobs=-1),
        XGBClassifier(random_state=RS, n_jobs=-1),
        MultiOutputClassifier(LGBMClassifier(is_unbalance=True, random_state=RS), n_jobs=-1)
    ],
    'tfidf__ngram_range': [(1,1), (1,2)]})

    models = [
        'logreg1', 'logreg2', 'sgd1', 'sgd2', 'svm1', 'svm2', 'knn1', 'knn2', 'rf1', 'rf2',
        'xgb1', 'xgb2', 'lgbm1', 'lgbm2'
    ]

    scores = pd.DataFrame()
    for model, params in tqdm.tqdm(zip(models, grid), total=len(models)):
        try:
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            machine_learning = score(y_test, y_pred, model)
            scores = pd.concat([scores, machine_learning])
        except Exception as e:
            print('uh oh: ' + str(e))

    print(scores)

def test_models2():
    with open('full_df.pkl', 'rb') as f:
        df = pickle.load(f)
    tags = df[['structural', 'lab', 'fe', 'rdd', 'diff-in-diff', 'iv', 'event-studies', 'na']]
    X_train, X_test, y_train, y_test = train_test_split(df['text'], tags, train_size=0.8, stratify=None, random_state=RS)

    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ClfSwitcher())])

    grid = ParameterGrid({
    'clf__estimator': [
        XGBClassifier(random_state=RS, n_jobs=-1),
        MultiOutputClassifier(LGBMClassifier(is_unbalance=True, random_state=RS), n_jobs=-1)
    ],
    'tfidf__ngram_range': [(1,1), (1,2), (1,3)]})

    models = [
        'xgb1', 'xgb2', 'xgb3', 'lgbm1', 'lgbm2', 'lgbm3'
    ]

    scores = pd.DataFrame()
    for model, params in tqdm.tqdm(zip(models, grid), total=len(models)):
        try:
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            machine_learning = score(y_test, y_pred, model)
            scores = pd.concat([scores, machine_learning])
        except Exception as e:
            print('uh oh: ' + str(e))

    print(scores)

def parameter_tune_lgbm():
    with open('full_df.pkl', 'rb') as f:
        df = pickle.load(f)
    tags = df[['structural', 'lab', 'fe', 'rdd', 'diff-in-diff', 'iv', 'event-studies', 'na']]

    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultiOutputClassifier(LGBMClassifier(is_unbalance=True, random_state=RS), n_jobs=-1))])

    param_grid = {
        'clf__estimator__boosting_type': ['gbdt', 'dart'],
        'clf__estimator__random_state': [5685],
        'clf__estimator__num_leaves': [16, 31, 40, 50],
        'clf__estimator__max_bin': [255, 500],
        'clf__estimator__learning_rate': [0.01, 0.1],
        'clf__estimator__n_estimators': [100, 200]
    }

    grid = GridSearchCV(pipeline, param_grid, verbose=2, n_jobs=-1)
    grid.fit(df['text'], tags)
    print(grid.best_params_)
    print(grid.best_score_)


def run_model():
    with open('full_df.pkl', 'rb') as f:
        df = pickle.load(f)
    tags = df[['structural', 'lab', 'afe', 'ffe', 'nfe', 'rdd', 'diff-in-diff', 'iv', 'event-studies', 'na']]
    X_train, X_test, y_train, y_test = train_test_split(df['text'], tags, train_size=0.8, stratify=None, random_state=RS)
    
    pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('clf', ClfSwitcher())])
    params = {
        'clf__estimator': MultiOutputClassifier(LGBMClassifier(is_unbalance=True, random_state=RS), n_jobs=-1),
        'tfidf__ngram_range': (1,2)
    }

    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred = pd.DataFrame(y_pred, index=X_test.index, columns=['structural2', 'lab2', 'afe2', 'ffe2', 'nfe2', 'rdd2', 'diff-in-diff2', 'iv2', 'event-studies2', 'na2'])
    out = pd.merge(y_test, y_pred, left_index=True, right_index=True)
    return out

if __name__ == '__main__':
    # make_full_df()
    # test_models2()
    parameter_tune_lgbm()