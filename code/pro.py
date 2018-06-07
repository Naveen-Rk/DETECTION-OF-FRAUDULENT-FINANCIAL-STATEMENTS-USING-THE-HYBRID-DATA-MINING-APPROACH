import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import os
import h2o
import numpy as np

data = reviews = pd.read_csv('input/creditcard.csv')
data['Class'] = data['Class'].astype('category')

cols = list(data.columns)
data.groupby('Class')['Class'].count()
---++h2o.init()
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


from scipy import stats
variables = cols[:-1]
keep = []
p_value_alpha = 0.05 #defult p-value for statistical significancekmik

for variable in variables:
    fraud_v = df[variable][df.Class == 1]
    not_fraud_v = df[variable][df.Class == 0].sample(len(fraud_v))
    p_value = stats.ttest_ind(not_fraud_v, fraud_v).pvalue
    if p_value >= p_value_alpha:
        print("Distributions are equal. Discard {} variable".format(variable))
    else:
        print("Distributions are different. Keep {} variable".format(variable))
        keep.append(variable)


#modeling
from h2o.estimators.random_forest import H2ORandomForestEstimator
train, valid = creditcard.split_frame(ratios=[0.7])
response_var = 'Class'
features = [col for col in cols if col != response_var]
naive_rf_model = H2ORandomForestEstimator()
naive_rf_model.train(x=features, y=response_var, training_frame=train, validation_frame=valid)
performance_train = naive_rf_model.model_performance(train=True)

print(performance_train)
