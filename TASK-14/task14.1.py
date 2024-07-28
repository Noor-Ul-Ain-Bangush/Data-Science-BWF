#using normal equation:
import numpy as np

# Generating random linear-looking data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Adding x0 = 1 to each instance
X_b = np.c_[np.ones((100, 1)), X]

# Calculating the best theta using the Normal Equation
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Theta calculated from Normal Equation:", theta_best)

# Making predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print("Predictions:", y_predict)

# Plotting the results
import matplotlib.pyplot as plt

plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.", label="Data Points")
plt.axis([0, 2, 0, 15])
plt.legend()
plt.show()


#using scikit-learn
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)

y_predict = lin_reg.predict(X_new)
print("Predictions:", y_predict)


#Gradient Descent Implementation
eta = 0.1  # Learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)  # Random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print("Theta calculated from Gradient Descent:", theta)


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures

# Generating polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Training the model
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

print("Intercept:", lin_reg.intercept_)
print("Coefficients:", lin_reg.coef_)

# Making predictions
X_new_poly = poly_features.transform(X_new)
y_predict = lin_reg.predict(X_new_poly)

print("Predictions:", y_predict)



#Ridge Regression
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
print("Intercept:", ridge_reg.intercept_)
print("Coefficients:", ridge_reg.coef_)

y_predict = ridge_reg.predict(X_new)
print("Predictions:", y_predict)


#Lasso Regression
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
print("Intercept:", lasso_reg.intercept_)
print("Coefficients:", lasso_reg.coef_)

y_predict = lasso_reg.predict(X_new)
print("Predictions:", y_predict)


#Elastic Net
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print("Intercept:", elastic_net.intercept_)
print("Coefficients:", elastic_net.coef_)

y_predict = elastic_net.predict(X_new)
print("Predictions:", y_predict)



