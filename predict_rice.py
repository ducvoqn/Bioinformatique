import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def cleanText(df, text):
    df.loc[:, text] = df[text].apply(lambda x: str.lower(x))  # Converting to lower case
    df.loc[:, text] = df[text].apply(lambda x: re.sub('go', ' go', x))
    df.loc[:, text] = df[text].apply(lambda x: re.sub('po', ' po', x))
    df.loc[:, text] = df[text].apply(lambda x: re.sub('to', ' to', x))
    df.loc[:, text] = df[text].apply(lambda x: re.sub('\\n', '', x))  # Remove \n
    df.loc[:, text] = df[text].apply(lambda x: re.sub('\W', ' ', x))  # Remove special character
    df.loc[:, text] = df[text].apply(lambda x: re.sub('\s+[a-zA-Z]\s+', ' ', x))  # Remove all single characters
    df.loc[:, text] = df[text].apply(lambda x: re.sub('\s+', ' ', x))  # Substitute multiple space with 1 space
    df.loc[:, text] = df[text].apply(lambda x: re.sub('quality of panicle', '', x))  # Remove redundant term
    return df


# Configure to display all columns in the console
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)

# Read data
df = pd.read_csv('ricedb.csv')

# Clean text
df = cleanText(df, 'Ontology')
df = cleanText(df, 'Role')
df = df[~(df['Role'].str.contains("panicle type", case=False))]

# Categorize Role by ID
df['role_id'], labels_text = df['Role'].factorize()

# Show proportion of categories
fig = plt.figure(figsize=(10, 8))
df.groupby('Role').Ontology.count().plot.bar(ylim=0)
plt.show()

# Convert text to numeric by tf-idf
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, ngram_range=(1, 2),
                        stop_words='english')

features = tfidf.fit_transform(df.Ontology).toarray()
features = pd.DataFrame(features)
labels = df.role_id

print(features.head(10))

# Find 2 terms that are most related to the category
N = 2
role_id_df = df[['Role', 'role_id']].sort_values('role_id')
role_to_id = dict(role_id_df.values)  # Create Dictionary
id_to_role = dict(role_id_df[['role_id', 'Role']].values)

for Role, role_id in role_to_id.items():
    features_chi2 = chi2(features, labels == role_id)
    indices = np.argsort(features_chi2[0])
    features_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in features_names if len(v.split(' ')) == 1]
    bigrams = [v for v in features_names if len(v.split(' ')) == 2]
    print("# '{}':".format(Role))
    print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
    print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# Evaluate different Models with K cross validation % F1-score
models = [
    RandomForestClassifier(n_estimators=500, oob_score=True),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression()
]

CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels,
                                 scoring='f1_micro', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx',
                                       'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
print("\n", cv_df.groupby('model_name').accuracy.mean())


model = MultinomialNB()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.25, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Confusion matrix with heat map
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=df.Role.unique(), yticklabels=df.Role.unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(classification_report(y_test, y_pred, target_names=df['Role'].unique()))

# # Improve performance with Predicted probabilities
# Class Prediction with probabilities
calibrated_svc = CalibratedClassifierCV(base_estimator=model, cv="prefit", method='sigmoid')
calibrated_svc.fit(X_train, y_train)
label_predicted = pd.DataFrame(calibrated_svc.predict(X_test))
prob_predicted = pd.DataFrame(calibrated_svc.predict_proba(X_test))
score = calibrated_svc.score(X_test, y_test)

conf_mat1 = confusion_matrix(label_predicted, y_test)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat1, annot=True, fmt='d',
            xticklabels=df.Role.unique(), yticklabels=df.Role.unique())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

classes2 = ['number of panicle on plant', 'grain quality', 'branch quality',
            'inflorescene']

prob_predicted.columns = classes2
print(prob_predicted.head(10))
print(confusion_matrix(y_test, label_predicted))
print('\nAccuracy score:', score)


# # # Prediction on new data

# # Unknown-label data
# test_data = pd.read_csv('list_genes.csv')
# test_data = cleanText(test_data, 'Ontology')
# test_data = cleanText(test_data, 'Rap ID')
# feature_test = tfidf.transform(test_data.Ontology).toarray()
# example = pd.DataFrame(feature_test)
#
# # Classes
# y0 = (y_train == 0) + 0
# y1 = (y_train == 1) + 0
# y2 = (y_train == 2) + 0
# y3 = (y_train == 3) + 0
#
# # Train Models
# fit0 = MultinomialNB()
# fit1 = MultinomialNB()
# fit2 = MultinomialNB()
# fit3 = MultinomialNB()
#
# fit0.fit(X_train, y0)
# fit1.fit(X_train, y1)
# fit2.fit(X_train, y2)
# fit3.fit(X_train, y3)

# calibrated_svc0 = CalibratedClassifierCV(base_estimator=fit0, cv="prefit", method='sigmoid')
# calibrated_svc1 = CalibratedClassifierCV(base_estimator=fit1, cv="prefit", method='sigmoid')
# calibrated_svc2 = CalibratedClassifierCV(base_estimator=fit2, cv="prefit", method='sigmoid')
# calibrated_svc3 = CalibratedClassifierCV(base_estimator=fit3, cv="prefit", method='sigmoid')
#
# calibrated_svc0.fit(X_train, y0)
# calibrated_svc1.fit(X_train, y1)
# calibrated_svc2.fit(X_train, y2)
# calibrated_svc3.fit(X_train, y3)
#
# pred = pd.DataFrame()
# pred[0] = calibrated_svc0.predict_proba(example)[:, 1].tolist()
# pred[1] = calibrated_svc1.predict_proba(example)[:, 1].tolist()
# pred[2] = calibrated_svc2.predict_proba(example)[:, 1].tolist()
# pred[3] = calibrated_svc3.predict_proba(example)[:, 1].tolist()
# pred[4] = 1 - pred[0] - pred[1] - pred[2] - pred[3]


# # Predict new data
# predicted = calibrated_svc.predict_proba(example)

# classes1 = ['number of panicle on plant', 'grain quality', 'branch quality',
#             'inflorescene', 'others']
# pred.columns = classes1
# test_data['Role'] = pred.idxmax(axis=1)
# pred = pred.multiply(100)
# print(test_data.head(10))
# print(pred.head(10))
