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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



# --------------------- importing dataset -----------------------

train_dataset = pd.read_csv('NSL-KDD__BinaryClass_Classification_Dataset.csv')
test_dataset = pd.read_csv('KDDTest+.csv')



# -------------------- taking care of inconsistent data -----------------

train_dataset = train_dataset.replace([np.inf, -np.inf], np.nan) 
train_dataset = train_dataset.dropna()
train_dataset = train_dataset.drop_duplicates()
print('No of null values : ', train_dataset.isnull().sum().sum())


test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan) 
test_dataset = test_dataset.dropna()
test_dataset = test_dataset.drop_duplicates()
print('No of null values : ', test_dataset.isnull().sum().sum())



# ---------- generalizing different types of attacks in the test set as Attack -------

test_dataset = test_dataset.rename(columns = {'class' : 'class_name'})
attack_list = test_dataset['class_name'].unique().tolist()
attack_list.remove('normal')
test_dataset = test_dataset.replace(attack_list, 'attack')



# --------------- creating dependent variable vector ----------------

y_train = train_dataset.iloc[:, -2].values
y_test = test_dataset.iloc[:, -2].values

print(np.asarray(np.unique(y_train, return_counts=True)))
print(np.asarray(np.unique(y_test, return_counts=True)))



# --------------- onehotencoding the categorical variables with dummy variables ----------------

train_dataset.drop(train_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
train_dataset = pd.get_dummies(train_dataset)
test_dataset.drop(test_dataset.iloc[:, [41, 42]], inplace = True, axis = 1) 
test_dataset = pd.get_dummies(test_dataset)



# ----- inserting columns of zeroes for categorical variables that are not common in train & test set -------

train_dataset.insert(62, "service_hostnames", 0)
train_dataset.insert(77, "service_netstat", 0)
train_dataset.insert(83, "service_pop_2", 0)

test_dataset.insert(87, "service_red_i", 0)

train_dataset.insert(98, "service_tftp_u", 0)

test_dataset.insert(101, "service_urh_i", 0)



# ----------------- creating matrix of features ----------------

X_train = train_dataset.iloc[:, :].values
X_test = test_dataset.iloc[:, :].values



# -------------- encoding categorical (dependent) variable -----------------

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

print(np.asarray(np.unique(y_train, return_counts=True)))
print(np.asarray(np.unique(y_test, return_counts=True)))



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
auc_scores = []
roc_auc_scores = []
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
    precision_scores.append(precision_score(y_validation, y_pred, average = 'binary'))
    recall_scores.append(recall_score(y_validation, y_pred, average = 'binary'))
    f1_scores.append(f1_score(y_validation, y_pred, average = 'binary'))
    
    
    # ---------------------- calculating value for auc -------------------------
    fpr, tpr, _ = roc_curve(y_validation, y_pred)
    auc_scores.append(auc(fpr, tpr))
    
    roc_auc_scores.append(roc_auc_score(y_validation, y_pred))
    
    
    # ---------------------- making confusion matrix -----------------------------
    cm = confusion_matrix(y_validation, y_pred)
    
    
    # --------------------- performance evaluation using the confusion matrix ---------------
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    
    
    true_negative_rates.append(tn / (fp + tn))
    false_positive_rates.append(fp / (fp + tn))
    false_negative_rates.append(fn / (fn + tp))
    false_alarm_rates.append((fp + fn) / (fp + fn + tp + tn))



# ------------------------ printing results of performance evaluation -------------------
    
print('================== Validation Set =====================')

print('Accuracy Score : ', np.mean(accuracy_scores))

print('Precision Score : ', np.mean(precision_scores))

print('Recall Score : ', np.mean(recall_scores))

print('F1 Score : ', np.mean(f1_scores)) 

print('AUC Score : ', np.mean(auc_scores))

print('ROC AUC Score : ', np.mean(roc_auc_scores))

print('Specificity or True Negative Rate : ', np.mean(true_negative_rates))

print('False Positive Rate : ', np.mean(false_positive_rates))

print('False Negative Rate : ', np.mean(false_negative_rates))

print('False Alarm Rate : ', np.mean(false_alarm_rates))   
    


# ------------------ predicting the test set result --------------------
  
y_pred = classifier.predict(X_test)



# --------------- making confusion matrix -----------------

cm = confusion_matrix(y_test, y_pred)



# ---------------- performance evaluation using confusion matrix -----------------

tn = cm[0][0]
fn = cm[1][0]
tp = cm[1][1]
fp = cm[0][1]



# ------------------ calculating values for each evaluator ------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average = 'binary')
recall = recall_score(y_test, y_pred, average = 'binary')
f1 = f1_score(y_test, y_pred, average = 'binary')

fpr, tpr, _ = roc_curve(y_test, y_pred)
auc_score = auc(fpr, tpr)

roc_auc = roc_auc_score(y_test, y_pred)

TNR = tn / (fp + tn)
FPR = fp / (fp + tn)
FNR = fn / (fn + tp)
FAR = (fp + fn) / (fp + fn + tp + tn)



# --------------- printing results of performance evaluation --------------------

print('================ Test Set ================')

print('Accuracy Score : ', accuracy)

print('Precision Score : ', precision)

print('Recall Score : ', recall)

print('F1 Score : ', f1)

print('AUC Score : ', auc_score)

print('ROC AUC Score : ', roc_auc)

print('Specificity or True Negative Rate : ', TNR)

print('False Positive Rate : ', FPR)

print('False Negative Rate : ', FNR)

print('False Alarm Rate : ', FAR)








