
"""pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with structured
(tabular, multidimensional, potentially heterogeneous) and time series data both easy and intuitive.
It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python"""
import pandas as pd

"""matplotlib strives to produce publication quality 2D graphics
for interactive graphing, scientific publishing, user interface
development and web application servers targeting multiple user
interfaces and hardcopy output formats. There is a pylab mode which emulates matlab graphics. """
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

"""Seaborn is a library for making attractive and informative statistical graphics in Python"""
import seaborn as sns

import os
"""H2O is a Java-based software for data modeling and general computing.
The H2O software is many things, but the primary purpose of H2O is as a distributed (many machines),
parallel (many CPUs), in memory (several hundred GBs Xmx) processing engine."""
import h2o

"""NumPy is a general-purpose array-processing package designed to efficiently manipulate large multi-dimensional
 arrays of arbitrary records without sacrificing too much speed for small multi-dimensional arrays."""
import numpy as np

data = reviews = pd.read_csv('input/creditcard.csv')
data['Class'] = data['Class'].astype('category')
cols = list(data.columns)


data.groupby('Class')['Class'].count()




h2o.init()

creditcard = h2o.upload_file(path ='input/creditcard.csv')

creditcard['Class'] = creditcard['Class'].asfactor()

df = creditcard.as_data_frame()

cols = list(df.columns)


#Checking for null values
df.isnull().sum()

#As transactions with Amount == 0 are impossible, let's check if there any in the dataset.
df.query('Amount == 0').groupby('Class').count()

df.query('Amount != 0').to_csv('creditcard_amount_positive.csv')
creditcard = h2o.upload_file(path ='creditcard_amount_positive.csv')
creditcard['Class'] = creditcard['Class'].asfactor()
df = creditcard.as_data_frame()



num_cols = len(cols[:-1])
plt.figure(figsize=(12,num_cols*4))
gs = gridspec.GridSpec(num_cols, 1)
for i, cn in enumerate(df[cols[:-1]]):
    ax = plt.subplot(gs[i])
    sns.distplot(df[cn][df.Class == 1], bins=50)
    sns.distplot(df[cn][df.Class == 0], bins=50)
    ax.set_xlabel('')
    ax.set_title('histogram of feature: ' + str(cn))
plt.show()

"""

Looking the charts we can guess that variables V13, V15, V20, V22, V23, V24, V25, V26 have similar curves between fraud and no fraud, maybe we can remove then because is dificult to find a treshold to separate fraud and no fraud."""

from scipy import stats
variables = cols[:-1]
keep = []
p_value_alpha = 0.05 #defult p-value for statistical significance

for variable in variables:
    fraud_v = df[variable][df.Class == 1]
    not_fraud_v = df[variable][df.Class == 0].sample(len(fraud_v))
    p_value = stats.ttest_ind(not_fraud_v, fraud_v).pvalue
    if p_value >= p_value_alpha:
        print("Distributions are equal. Discard {} variable".format(variable))
    else:
        print("Distributions are diferent. Keep {} variable".format(variable))
        keep.append(variable)


#modeling
from h2o.estimators.random_forest import H2ORandomForestEstimator
train, valid = creditcard.split_frame(ratios=[0.7])
response_var = 'Class'
features = [col for col in cols if col != response_var]
naive_rf_model = H2ORandomForestEstimator()
naive_rf_model.train(x=features, y=response_var, training_frame=train, validation_frame=valid)
performance_train = naive_rf_model.model_performance(train=True)


# for metrics
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


preds = naive_rf_model.predict(train)
cm = confusion_matrix(train.as_data_frame()['Class'], preds.as_data_frame()['predict'])
plot_confusion_matrix(cm, ['Non-Fraud', 'Fraud'], False)


fpr, tpr, threshold = roc_curve(train.as_data_frame()['Class'], preds.as_data_frame()['predict'])
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.04, 1.0])
plt.ylim([-0.04, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()



predictions = naive_rf_model.predict(valid)
cm = confusion_matrix(valid.as_data_frame()['Class'], predictions.as_data_frame()['predict'])
plot_confusion_matrix(cm, ['Non-Fraud', 'Fraud'], False)



#Confusion Matrix

fpr, tpr, threshold = roc_curve(valid.as_data_frame()['Class'], predictions.as_data_frame()['predict'])
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.04, 1.0])
plt.ylim([-0.04, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
