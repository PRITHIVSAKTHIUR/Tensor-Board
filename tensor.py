import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


# Load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Drop duplicate columns
test1 = test.drop(['Stay', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)
train1 = train.drop(['case_id', 'patientid', 'Hospital_region_code', 'Ward_Facility_Code'], axis=1)

# Splitting train data for Naive Bayes and XGBoost
X1 = train1.drop('Stay', axis=1)
y1 = train1['Stay']
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.20, random_state=100)

# Define XGBoost classifier
classifier_xgb = xgboost.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=800,
                                  objective='multi:softmax', reg_alpha=0.5, reg_lambda=1.5,
                                  booster='gbtree', n_jobs=4, min_child_weight=2, base_score=0.75)

# Train XGBoost model
model_xgb = classifier_xgb.fit(X_train, y_train)

# Predictions
prediction_xgb = model_xgb.predict(X_test)
acc_score_xgb = accuracy_score(prediction_xgb, y_test)
print("Accuracy:", acc_score_xgb*100)

# Set up TensorBoard
log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train XGBoost model with TensorBoard logging
model_xgb_with_tb = classifier_xgb.fit(X_train, y_train, callbacks=[tensorboard_callback])

# Run TensorBoard
os.system('tensorboard --logdir logs/fit')
