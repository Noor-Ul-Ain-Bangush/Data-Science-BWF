# Import necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor

# Load the dataset
housing = fetch_california_housing()

# Create a DataFrame
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)
housing_df['Target'] = housing.target

# Display the first few rows of the DataFrame
print("First few rows of the California housing dataset:")
print(housing_df.head())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# Check the shapes of the training and testing sets
print("\nShapes of training and testing sets:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)

# Train a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Make predictions with the linear regression model
y_pred = lin_reg.predict(X_test)

# Evaluate the linear regression model
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error of Linear Regression model:", mse)

# Perform cross-validation
scores = cross_val_score(lin_reg, housing.data, housing.target, scoring='neg_mean_squared_error', cv=5)

# Compute the mean and standard deviation of the cross-validation scores
rmse_scores = -scores
print("\nCross-Validation Scores (Linear Regression):", rmse_scores)
print("Mean:", rmse_scores.mean())
print("Standard Deviation:", rmse_scores.std())

# Create a pipeline with a standard scaler and linear regression model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lin_reg', LinearRegression())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions and evaluate the pipeline
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error with Pipeline (Linear Regression):", mse)

# Define the parameter grid for hyperparameter tuning with SVR
param_grid = [
    {'svr__kernel': ['linear'], 'svr__C': [1, 10, 100]},
    {'svr__kernel': ['rbf'], 'svr__C': [1, 10, 100], 'svr__gamma': [0.01, 0.1, 1]}
]

# Create a pipeline with a standard scaler and SVR model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Perform grid search for hyperparameter tuning
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Display the best parameters and score from the grid search
print("\nBest Parameters (Grid Search with SVR):", grid_search.best_params_)
print("Best Score (Grid Search with SVR):", grid_search.best_score_)

# Create a pipeline with a standard scaler and SGD regressor for out-of-core learning
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('sgd_reg', SGDRegressor(max_iter=1000, tol=1e-3, random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Make predictions and evaluate the pipeline
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error with Pipeline (SGD Regressor):", mse)
