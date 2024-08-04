import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv(r"C:\Users\Noor Ul Ain\OneDrive - University of Engineering and Technology Taxila\Desktop\ML\ML\LAB\ML_LAB12\LAB12_DATASET/spam_assassin.csv")

# Display the first few rows of the dataset
print(dataset.head())

# Extract features and target
data, target = dataset.text, dataset.target

# Display shapes
print(data.shape)
print(target.shape)

# Display dataset info
print(dataset.info())

# Split the dataset into training and testing sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, test_index in sss.split(data, target):
    train_X, test_X = data.iloc[train_index], data.iloc[test_index]
    train_y, test_y = target.iloc[train_index], target.iloc[test_index]

# Vectorize the text data
vect = TfidfVectorizer(min_df=5, ngram_range=(1, 3)).fit(train_X)
X_train_vectorized = vect.transform(train_X)

# Add additional features
def add_feature(X, feature_to_add):
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

add_length = train_X.str.len()
add_digits = train_X.str.count(r'\d')
add_dollars = train_X.str.count(r'\$')
add_characters = train_X.str.count(r'\W')

X_train_transformed = add_feature(X_train_vectorized, [add_length, add_digits, add_dollars, add_characters])

add_length_t = test_X.str.len()
add_digits_t = test_X.str.count(r'\d')
add_dollars_t = test_X.str.count(r'\$')
add_characters_t = test_X.str.count(r'\W')

X_test_transformed = add_feature(vect.transform(test_X), [add_length_t, add_digits_t, add_dollars_t, add_characters_t])

# Train and evaluate RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train_transformed, train_y, cv=3, method='predict_proba')

# Hyperparameter tuning with RandomizedSearchCV
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestClassifier()
random_search = RandomizedSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
random_search.fit(X_train_transformed, train_y)

# Display best parameters and estimator
print("Best parameters:", random_search.best_params_)
print("Best estimator:", random_search.best_estimator_)

# Fit final model and make predictions
final_model = random_search.best_estimator_
final_model.fit(X_train_transformed, train_y)
predictions = final_model.predict(X_test_transformed)

# Evaluate accuracy
print("Accuracy score:", accuracy_score(predictions, test_y))
