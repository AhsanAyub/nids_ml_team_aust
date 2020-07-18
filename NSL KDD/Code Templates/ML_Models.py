#%%
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

#%%
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

#%%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
classifier = LDA()

#%%
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
classifier = QDA()

#%%
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()

#%%
from sklearn.svm import LinearSVC
classifier = LinearSVC()

#%%
from sklearn.svm import SVC
classifier = SVC()

#%%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#%%
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier()

#%%
from sklearn.ensemble import AdaBoostClassifier
classifier = AdaBoostClassifier()

#%%
# pip install xgboost
import xgboost
from xgboost import XGBClassifier
classifier = XGBClassifier()

#%%
# pip install catboost
import catboost
from catboost import CatBoostClassifier
classifier = CatBoostClassifier()

#%%
# pip install lightgbm
import lightgbm
from lightgbm import LGBMClassifier
classifier = LGBMClassifier()

#%%
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
classifier = HistGradientBoostingClassifier()

#%%
from sklearn.linear_model import RidgeClassifier
classifier = RidgeClassifier()

#%%
from sklearn.linear_model import SGDClassifier
classifier = SGDClassifier()

#%%
from sklearn.linear_model import Perceptron
classifier = Perceptron()

#%%
from sklearn.linear_model import PassiveAggressiveClassifier
classifier = PassiveAggressiveClassifier()

#%%
from sklearn.neighbors import NearestCentroid
classifier = NearestCentroid()

#%%
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()

#%%
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()

#%%
from sklearn.ensemble import BaggingClassifier
classifier = BaggingClassifier()

#%%
from sklearn.ensemble import ExtraTreesClassifier
classifier = ExtraTreesClassifier()

#%%
# pip install logitboost
from logitboost import LogitBoost
classifier = LogitBoost()

#%%
# pip install rotation_forest
from rotation_forest import RotationTreeClassifier
classifier = RotationTreeClassifier()

#%%
# pip install rotation_forest
from rotation_forest import RotationForestClassifier
classifier = RotationForestClassifier()

#%%
from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()

#%%































