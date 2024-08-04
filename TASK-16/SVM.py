import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.metrics import classification_report


# Function to plot decision boundaries
def plot_decision_boundary(clf, X, y, title):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contour(xx, yy, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    plt.title(title)


# 1. Binary Classification with Linear SVM
plt.subplot(2, 2, 1)
X1 = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y1 = np.array([0, 0, 0, 1, 1, 1])  # Binary labels
clf1 = svm.SVC(kernel='linear')
clf1.fit(X1, y1)
plot_decision_boundary(clf1, X1, y1, "Linear SVM Binary Classification")

# 2. Non-Linear Classification with RBF Kernel
plt.subplot(2, 2, 2)
X2, y2 = make_circles(n_samples=100, noise=0.1, factor=0.4)
clf2 = svm.SVC(kernel='rbf')
clf2.fit(X2, y2)
plot_decision_boundary(clf2, X2, y2, "Non-Linear SVM (RBF Kernel)")

# 3. Multiclass Classification with Iris Dataset
plt.subplot(2, 2, 3)
iris = datasets.load_iris()
X3 = iris.data[:, :2]  # Use only the first two features
y3 = iris.target
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3, random_state=42)
clf3 = svm.SVC(kernel='linear', decision_function_shape='ovr')
clf3.fit(X_train, y_train)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='autumn')
plt.title("Iris Dataset Multiclass SVM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

# 4. SVM for Image Classification (Digits Dataset)
plt.subplot(2, 2, 4)
digits = datasets.load_digits()
X4 = digits.data
y4 = digits.target
X_train, X_test, y_train, y_test = train_test_split(X4, y4, test_size=0.5, random_state=42)
clf4 = svm.SVC(gamma=0.001)
clf4.fit(X_train, y_train)
y_pred = clf4.predict(X_test)
print(classification_report(y_test, y_pred))

plt.tight_layout()
plt.show()