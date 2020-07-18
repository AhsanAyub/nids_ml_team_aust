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

train_dataset = pd.read_csv('NSL-KDD__Multiclass_Classification_Dataset.csv')
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



# --------------- generalising different types of attacks in test set --------------------------

DoS = ['apache2','mailbomb','neptune','teardrop','smurf','pod','back','land','processtable']
test_dataset = test_dataset.replace(to_replace = DoS, value = 'DoS')

U2R = ['httptunnel','ps','xterm','sqlattack','rootkit','buffer_overflow','loadmodule','perl']
test_dataset = test_dataset.replace(to_replace = U2R, value = 'U2R')

R2L = ['udpstorm','worm','snmpgetattack','sendmail','named','snmpguess','xsnoop','xlock','warezclient','guess_passwd','ftp_write','multihop','imap','phf','warezmaster','spy']
test_dataset = test_dataset.replace(to_replace = R2L, value = 'R2L')

Probe = ['mscan','saint','ipsweep','portsweep','nmap','satan']
test_dataset = test_dataset.replace(to_replace = Probe, value = 'Probe')



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

train_dataset.insert(78, "service_nnsp", 0)
train_dataset.insert(83, "service_pop_2", 0)
train_dataset.insert(85, "service_printer", 0)
train_dataset.insert(87, "service_remote_job", 0)
train_dataset.insert(88, "service_rje", 0)
train_dataset.insert(89, "service_shell", 0)
train_dataset.insert(97, "service_tftp_u", 0)



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

#classifier = """#########################################"""


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
    for i in range(0, 5):
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

for i in range(0, 5):    
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












