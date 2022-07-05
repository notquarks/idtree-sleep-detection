import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

col_names = ['sr', 'rr', 't', 'lm', 'bo', 'rem', 'sh', 'hr', 'sl']
col_names = ['Snoring Range', 'Respiration Rate', 'Temp', 'Limb Movement',
             'Blood Oxygen', 'REM', 'Sleep Hour', 'Heart Rate', 'sl']
sleep = pd.read_csv("sleep.csv", header=None, names=col_names)
sleep.head(n=200)
feature_cols = ['Snoring Range', 'Respiration Rate', 'Temp',
                'Limb Movement', 'Blood Oxygen', 'REM', 'Sleep Hour', 'Heart Rate']
X = sleep[feature_cols]  # Features
y = sleep.sl  # Target variable
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)
X = sleep[feature_cols]  # Features
y = sleep.sl  # Target variable
# Create Decision Tree classifer
clf = DecisionTreeClassifier()
# Train Decision Tree
clf = clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
print("Akurasi:", metrics.accuracy_score(y_test, y_pred))
joblib.dump(clf, "clf.pkl")
