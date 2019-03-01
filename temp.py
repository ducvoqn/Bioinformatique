import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, KFold

pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)

class PR(object):
    """Predict rice"""
    def __init__(self, data_path):
        self.raw = pd.read_csv(data_path)  # Original data set
        self.tfidf = TfidfVectorizer()
        self.cv_df = pd.DataFrame()
        # self.algorithm = algorithm

    def clean_text(self, df, column_name):
        df.loc[:, column_name] = df[column_name].apply(lambda x: str.lower(x))  # Converting to lower case
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('go', ' go', x))
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('po', ' po', x))
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('to', ' to', x))
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('\\n', '', x))  # Remove \n
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('\W', ' ', x))  # Remove special character
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('\s+[a-zA-Z]\s+', ' ', x))  # Remove all single characters
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('\s+', ' ', x))  # Substitute multiple space with 1 space
        df.loc[:, column_name] = df[column_name].apply(lambda x: re.sub('quality of panicle', '', x))  # Remove redundant term
        return df

    def preprocess_data(self, df):
        df = df[~(df[df.columns.values[1]].str.contains("panicle type", case=False))]
        df = self.clean_text(df, "Ontology")
        df = self.clean_text(df, "Role")
        df['role_id'] = df['Role'].factorize()[0]
        return df

    def display_distribution(self, df):
        df.groupby('Role').Ontology.count().plot.bar(x="Role", y="Number of candidates")
        plt.show()

    def fit_transform_data(self, feature):
        self.tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2),stop_words='english')
        tf_feature = pd.DataFrame(self.tfidf.fit_transform(feature).toarray())
        tf_feature.columns = self.tfidf.get_feature_names()
        return tf_feature

    def benchmark(self, df, no_folds):
        # PREPROCESS DATA
        df = self.preprocess_data(df)

        # TRANSFORM DATA
        features = self.fit_transform_data(df.Ontology)

        # BENCHMARK ALGORITHM
        labels = df.Role
        models = [
            RandomForestClassifier(),
            LinearSVC(),
            MultinomialNB(),
            LogisticRegression()
        ]
        entries = []
        for model in models:
            model_name = model.__class__.__name__
            accuracies = cross_val_score(model, features, labels, scoring='f1_micro', cv=no_folds)
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy))
        self.cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'f1_micro'])
        avg_score = self.cv_df.groupby('model_name').f1_micro.mean()
        print(avg_score)
        # self.plot_box()
        return self.cv_df

    def plot_box(self):
        sns.boxplot(x='model_name', y='f1_micro', data=self.cv_df)
        sns.stripplot(x='model_name', y='f1_micro', data=self.cv_df,
                      size=8, jitter=True, edgecolor="gray", linewidth=2)
        plt.show()

def predict_prob(model, X_train, y_train, X_test):
    pred = pd.DataFrame()
    models = [MultinomialNB(), MultinomialNB(), MultinomialNB(), MultinomialNB()]
    for i in range(0,4):
        model = models[i]
        pred[i] = model.fit(X_train, (y_train == i) + 0).predict_proba(X_test)[:,1].tolist()
    pred[4] = 1 - pred[0] - pred[1] - pred[2] - pred[3]
    pred = pred.multiply(100)
    return pred





a = PR('ricedb.csv')
b = PR('list_genes.csv')
df = a.raw
a.benchmark(df, 5)
tfidf = a.tfidf
X_test = tfidf.transform(b.raw.Ontology).toarray()
df = a.preprocess_data(df)
features = a.fit_transform_data(df.Ontology)
model = MultinomialNB()
pred = predict_prob(model, features, df.role_id, X_test)

