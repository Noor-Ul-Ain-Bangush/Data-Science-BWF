import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split

# Create a simple binary classification dataset
X1 = np.array([[1], [5], [1.5], [8], [1], [9], [7], [8.7], [2.3], [5.5], [7.7], [6.1]])
y1 = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])  # Binary labels (0 or 1)

# Create a more complex dataset (Iris dataset)
iris = datasets.load_iris()
X2 = iris.data[:, :2]  # Use only the first two features for simplicity
y2 = iris.target

# Convert the Iris target to binary for simplicity
y2 = np.where(y2 == 0, 0, 1)  # Binarize the targets

# Split the Iris dataset
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=42)

# Define linear SVM models
clf1 = svm.SVC(kernel='linear', C=1.0)
clf1.fit(X1, y1)

clf2 = svm.SVC(kernel='rbf', C=1.0)
clf2.fit(X_train, y_train)


# Plot the decision boundaries for both models
def plot_decision_boundary(clf, X, y, title):
    plt.figure()
    if X.shape[1] == 1:  # Check if there is only one feature
        # Create a meshgrid for one feature
        xlim = [X.min() - 1, X.max() + 1]
        xx = np.linspace(xlim[0], xlim[1], 100).reshape(-1, 1)
        Z = clf.decision_function(xx)
        plt.plot(xx, Z, color='k')  # Plot the decision function
        plt.scatter(X, y, s=50, cmap='autumn')
    else:
        plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        ax = plt.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_xlim()
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                             np.linspace(ylim[0], ylim[1], 100))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title(title)


# Plot linear SVM with binary classification dataset
plot_decision_boundary(clf1, X1, y1, "Linear SVM with Binary Classification Dataset")

# Plot non-linear SVM with Iris dataset
plot_decision_boundary(clf2, X_test, y_test, "Non-Linear SVM with Iris Dataset")

plt.show()