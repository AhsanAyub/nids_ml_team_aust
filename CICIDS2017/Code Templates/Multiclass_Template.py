# --------------------- importing libraries ---------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- importing class for label encoding ------------

from sklearn.preprocessing import LabelEncoder

# --------- importing method to split the dataset -------------------

from sklearn.model_selection import train_test_split

# ------------- importing class for feature scaling --------------------

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# ------------ importing class for cross validation --------------

from sklearn.model_selection import StratifiedKFold

# ------------ importing class for machine learning model ----------

"""###################################################""" 

# ----------- importing methods for performance evaluation ------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import auc
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve



# --------------------- importing dataset -----------------------

dataset = pd.read_csv('IDS2017__Multiclass_Classification_Dataset.csv')



# -------------------- taking care of inconsistent data -----------------

dataset = dataset.replace([np.inf, -np.inf], np.nan) 
dataset = dataset.dropna()
dataset = dataset.drop_duplicates()
print('No of null values : ', dataset.isnull().sum().sum())



# --------------- creating matrix of features and dependent variable vector ----------------

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(np.asarray(np.unique(y, return_counts=True)))



# -------------- encoding categorical (dependent) variable -----------------

le = LabelEncoder()
y = le.fit_transform(y)

print(np.asarray(np.unique(y, return_counts=True)))



# --------------- splitting into train set and test set ------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 43)



# --------------------------- feature scaling --------------------------------

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



# ---------------------- stratified k fold ---------------------------

cv = StratifiedKFold()

classifier = """#########################################"""


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
true_negative_rates = []
false_positive_rates = []
false_negative_rates = []
false_alarm_rates = []


for train, test in cv.split(X_train, y_train):
    
    
    # ------------------------ splitting the train set into NEW train set and validation set ------------------------
    X_train_fold, X_validation, y_train_fold, y_validation = X_train[train], X_train[test], y_train[train], y_train[test]
    
    
    # ------------------ fitting the classifier to the NEW train set ------------------------
    classifier.fit(X_train_fold, y_train_fold)
    
    
    # --------------------- predicting the validation set result ------------------------
    y_pred = classifier.predict(X_validation)
    
    
    # ------------------------ performance evaluation using methods from scikit learn --------------------------------
    accuracy_scores.append(accuracy_score(y_validation, y_pred))
    precision_scores.append(np.mean(precision_score(y_validation, y_pred, average = None)))
    recall_scores.append(np.mean(recall_score(y_validation, y_pred, average = None)))
    f1_scores.append(np.mean(f1_score(y_validation, y_pred, average = None)))
    
    
    # ---------------------- making confusion matrix -----------------------------
    cm = multilabel_confusion_matrix(y_validation, y_pred)
    
    
    # --------------------- performance evaluation using the confusion matrix ---------------
    for i in range(0, 8):
        tn = cm[i][0][0]
        fn = cm[i][1][0]
        tp = cm[i][1][1]
        fp = cm[i][0][1]
    
        TNR = []
        FPR = []
        FNR = []
        FAR = []
    
    
        # ---------------- calculating values for ecah class -----------------
        TNR.append(tn / (fp + tn))
        FPR.append(fp / (fp + tn))
        FNR.append(fn / (fn + tp))
        FAR.append((fp + fn) / (fp + fn + tp + tn))


    # ------------- getting average values (of all classes) for each fold (each validation set) ----------------    
    true_negative_rates.append(np.mean(TNR))
    false_positive_rates.append(np.mean(FPR))
    false_negative_rates.append(np.mean(FNR))
    false_alarm_rates.append(np.mean(FAR))



# ------------------------ printing results of performance evaluation -------------------
    
print('================== Validation Set =====================')

print('Accuracy Score : ', np.mean(accuracy_scores))

print('Precision Score : ', np.mean(precision_scores))

print('Recall Score : ', np.mean(recall_scores))

print('F1 Score : ', np.mean(f1_scores)) 

print('Specificity or True Negative Rate : ', np.mean(true_negative_rates))

print('False Positive Rate : ', np.mean(false_positive_rates))

print('False Negative Rate : ', np.mean(false_negative_rates))

print('False Alarm Rate : ', np.mean(false_alarm_rates))   
    


# ------------------ predicting the test set result --------------------
  
y_pred = classifier.predict(X_test)



# --------------- making confusion matrix -----------------

cm = multilabel_confusion_matrix(y_test, y_pred)



# ---------------- performance evaluation using confusion matrix -----------------

for i in range(0, 8):    
    tn = cm[i][0][0]
    fn = cm[i][1][0]
    tp = cm[i][1][1]
    fp = cm[i][0][1]
    
    
    TNR = []
    FPR = []
    FNR = []
    FAR = []
    
    
    # ----------- calculating values for each class ------------------
    TNR.append(tn / (fp + tn))
    FPR.append(fp / (fp + tn))
    FNR.append(fn / (fn + tp))
    FAR.append((fp + fn) / (fp + fn + tp + tn))



# --------------- printing results of performance evaluation --------------------

print('================ Test Set ================')

print('Accuracy Score : ', accuracy_score(y_test, y_pred))

print('Precision Score : ', np.mean(precision_score(y_test, y_pred, average = None)))

print('Recall Score : ', np.mean(recall_score(y_test, y_pred, average = None)))

print('F1 Score : ', np.mean(f1_score(y_test, y_pred, average = None)))

print('Specificity or True Negative Rate : ', np.mean(TNR))

print('False Positive Rate : ', np.mean(FPR))

print('False Negative Rate : ', np.mean(FNR))

print('False Alarm Rate : ', np.mean(FAR))















