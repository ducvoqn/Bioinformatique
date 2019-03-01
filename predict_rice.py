import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
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

# # Show proportion of categories
# fig = plt.figure(figsize=(10, 8))
# df.groupby('Role').Ontology.count().plot.bar(ylim=0)
# plt.show()

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

# for Role, role_id in role_to_id.items():
#     features_chi2 = chi2(features, labels == role_id)
#     indices = np.argsort(features_chi2[0])
#     features_names = np.array(tfidf.get_feature_names())[indices]
#     unigrams = [v for v in features_names if len(v.split(' ')) == 1]
#     bigrams = [v for v in features_names if len(v.split(' ')) == 2]
#     print("# '{}':".format(Role))
#     print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
#     print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

# Evaluate different Models
# models = [
#     RandomForestClassifier(),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression()
# ]
#
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#     model_name = model.__class__.__name__
#     accuracies = cross_val_score(model, features, labels,
#                                  scoring='f1_micro', cv=CV)
#     for fold_idx, accuracy in enumerate(accuracies):
#         entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx',
#                                        'f1_micro'])
#
# sns.boxplot(x='model_name', y='f1_micro', data=cv_df)
# sns.stripplot(x='model_name', y='f1_micro', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
# print(cv_df.groupby('model_name').f1_micro.mean())

kf = KFold(n_splits=5, random_state=0)
f1_scores = []
f1_scores_calibrated = []

for train_index, test_index in kf.split(features):
    model = MultinomialNB()
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    y_train = y_train.fillna(0)
    y_train = y_train.astype(np.int32)
    y_train = np.asarray(y_train)
    y_test = y_test.fillna(0)
    y_test = y_test.astype(np.int32)
    y_test = np.asarray(y_test)

    # X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
    #                                                                              test_size=0.25, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Classes
    y0 = (y_train == 0) + 0
    y1 = (y_train == 1) + 0
    y2 = (y_train == 2) + 0
    y3 = (y_train == 3) + 0

    # Train Models
    fit0 = MultinomialNB()
    fit1 = MultinomialNB()
    fit2 = MultinomialNB()
    fit3 = MultinomialNB()

    fit0.fit(X_train, y0)
    fit1.fit(X_train, y1)
    fit2.fit(X_train, y2)
    fit3.fit(X_train, y3)

    predz = pd.DataFrame()
    predz[0] = fit0.predict_proba(X_test)[:, 1].tolist()
    predz[1] = fit1.predict_proba(X_test)[:, 1].tolist()
    predz[2] = fit2.predict_proba(X_test)[:, 1].tolist()
    predz[3] = fit3.predict_proba(X_test)[:, 1].tolist()
    predz[4] = 1 - predz[0] - predz[1] - predz[2] - predz[3]

    # predz.columns = classes1
    y_predz = predz.idxmax(axis=1)

    f1_scores.append((f1_score(y_test, y_predz, average='micro')))
    cm = confusion_matrix(y_test, y_predz)

    # # Confusion matrix with heat map
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(cm, annot=True, fmt='d',
    #             xticklabels=df.Role.unique(), yticklabels=df.Role.unique())
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()

    calibrated_svc = CalibratedClassifierCV(base_estimator=model, cv="prefit", method='isotonic')
    calibrated_svc.fit(X_train, y_train)

    calibrated_svc0 = CalibratedClassifierCV(base_estimator=fit0, cv="prefit", method='isotonic')
    calibrated_svc1 = CalibratedClassifierCV(base_estimator=fit1, cv="prefit", method='isotonic')
    calibrated_svc2 = CalibratedClassifierCV(base_estimator=fit2, cv="prefit", method='isotonic')
    calibrated_svc3 = CalibratedClassifierCV(base_estimator=fit3, cv="prefit", method='isotonic')

    calibrated_svc0.fit(X_train, y0)
    calibrated_svc1.fit(X_train, y1)
    calibrated_svc2.fit(X_train, y2)
    calibrated_svc3.fit(X_train, y3)

    preds = pd.DataFrame()
    preds[0] = calibrated_svc0.predict_proba(X_test)[:, 1].tolist()
    preds[1] = calibrated_svc1.predict_proba(X_test)[:, 1].tolist()
    preds[2] = calibrated_svc2.predict_proba(X_test)[:, 1].tolist()
    preds[3] = calibrated_svc3.predict_proba(X_test)[:, 1].tolist()
    preds[4] = 1 - preds[0] - preds[1] - preds[2] - preds[3]

    y_preds = preds.idxmax(axis=1)
    f1_scores_calibrated.append((f1_score(y_test, y_preds, average='micro')))
    # cm_calibrated = confusion_matrix(y_test, y_preds)
    # # Confusion matrix with heat map
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.heatmap(cm_calibrated, annot=True, fmt='d',
    #             xticklabels=df.Role.unique(), yticklabels=df.Role.unique())
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    # plt.show()


print("Uncalibrated: ", np.mean(f1_scores))
print("Calibrated: ", np.mean(f1_scores_calibrated))
# print(classification_report(y_test, y_pred, target_names=df['Role'].unique()))



# # Predicted probabilities

test_data = pd.read_csv('list_genes.csv')
test_data = cleanText(test_data, 'Ontology')
test_data = cleanText(test_data, 'Rap ID')
feature_test = tfidf.transform(test_data.Ontology).toarray()
example = pd.DataFrame(feature_test)

classes1 = ['number of panicle on plant', 'grain quality', 'branch quality',
            'inflorescene', 'others']


# # Classes
y0 = (labels == 0) + 0
y1 = (labels == 1) + 0
y2 = (labels == 2) + 0
y3 = (labels == 3) + 0
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
#
# predz = pd.DataFrame()
# predz[0] = fit0.predict_proba(X_test)[:, 1].tolist()
# predz[1] = fit1.predict_proba(X_test)[:, 1].tolist()
# predz[2] = fit2.predict_proba(X_test)[:, 1].tolist()
# predz[3] = fit3.predict_proba(X_test)[:, 1].tolist()
# predz[4] = 1 - predz[0] - predz[1] - predz[2] - predz[3]
#
# # predz.columns = classes1
# y_predz = predz.idxmax(axis=1)
# print(classification_report(y_test, y_predz, target_names=df['Role'].unique()))

# Class Prediction with probabilities
# calibrated_svc = CalibratedClassifierCV(base_estimator=model, cv="prefit", method='sigmoid')
# calibrated_svc.fit(X_train, y_train)
#
# calibrated_svc0 = CalibratedClassifierCV(base_estimator=fit0, cv="prefit", method='sigmoid')
# calibrated_svc1 = CalibratedClassifierCV(base_estimator=fit1, cv="prefit", method='sigmoid')
# calibrated_svc2 = CalibratedClassifierCV(base_estimator=fit2, cv="prefit", method='sigmoid')
# calibrated_svc3 = CalibratedClassifierCV(base_estimator=fit3, cv="prefit", method='sigmoid')
#
calibrated_svc0.fit(features, y0)
calibrated_svc1.fit(features, y1)
calibrated_svc2.fit(features, y2)
calibrated_svc3.fit(features, y3)
#
# preds = pd.DataFrame()
# preds[0] = calibrated_svc0.predict_proba(X_test)[:, 1].tolist()
# preds[1] = calibrated_svc1.predict_proba(X_test)[:, 1].tolist()
# preds[2] = calibrated_svc2.predict_proba(X_test)[:, 1].tolist()
# preds[3] = calibrated_svc3.predict_proba(X_test)[:, 1].tolist()
# # preds[4] = 1 - preds[0] - preds[1] - preds[2] - preds[3]
#
# y_preds = preds.idxmax(axis=1)
# print(classification_report(y_test, y_preds, target_names=df['Role'].unique()))


pred = pd.DataFrame()
pred[0] = calibrated_svc0.predict_proba(example)[:, 1].tolist()
pred[1] = calibrated_svc1.predict_proba(example)[:, 1].tolist()
pred[2] = calibrated_svc2.predict_proba(example)[:, 1].tolist()
pred[3] = calibrated_svc3.predict_proba(example)[:, 1].tolist()
pred[4] = 1 - pred[0] - pred[1] - pred[2] - pred[3]

# label_predicted = pd.DataFrame(calibrated_svc.predict(X_test))
# prob_predicted = pd.DataFrame(calibrated_svc.predict_proba(X_test))
# score = calibrated_svc.score(X_test, y_test)

# # Predict new data
# predicted = calibrated_svc.predict_proba(example)


# conf_mat1 = confusion_matrix(label_predicted, y_test)
# fig, ax = plt.subplots(figsize=(10, 10))
# sns.heatmap(conf_mat1, annot=True, fmt='d',
#             xticklabels=df.Role.unique(), yticklabels=df.Role.unique())
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()



# prob_predicted.columns = classes2
# print(prob_predicted.head(10))
# print(confusion_matrix(y_test, label_predicted))
# print('\nAccuracy score:', score)

pred.columns = classes1
test_data['Role'] = pred.idxmax(axis=1)
pred = pred.multiply(100)
test_data['Probability'] = pred.max(axis=1)
print(test_data.head(10))
print(pred.head(50))
test_data.to_excel("result.xlsx", sheet_name='Sheet_name_1')

