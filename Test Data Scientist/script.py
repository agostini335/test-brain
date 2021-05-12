#%% LOADING AND PRE PROCESSING
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score


#data loading
df = pd.read_csv("data.csv")
print(df.head)

#shape - controllo dimensionalità
print(df.shape)
#info - controllo tipi
print(df.info())
#drop nan - rimozione righe con valori nulli
df.dropna(inplace = True)

#descrizione 
print(df.describe())
df['default'].sum()



# %% DATA SPLITTING
Y = df["default"]
X = df.drop(columns = ["default"])

train, test, train_labels, test_labels = train_test_split(X,
                                                          Y,
                                                        test_size=0.2)
 
# %% Classifier
gnb = GaussianNB()
lr = LogisticRegression()

# Trainining
gnb.fit(train, train_labels)
lr.fit(train, train_labels)

#compute predictions
preds_gnb = gnb.predict(test)
preds_lr = lr.predict(test)
preds_proba_lr = lr.predict_proba(test)

print(preds_lr[0:20])
print(preds_proba_lr[0:20])

# %% EVALUATION
print("Logistic Regression")
print("accuracy")
print(accuracy_score(test_labels, preds_lr))
print("precision")
print(precision_score(test_labels, preds_lr))
print("recall")
print(recall_score(test_labels, preds_lr))

print("\nNaive Bayes")
print("accuracy")
print(accuracy_score(test_labels, preds_gnb))
print("precision")
print(precision_score(test_labels, preds_gnb))
print("recall")
print(recall_score(test_labels, preds_gnb))



# %%
