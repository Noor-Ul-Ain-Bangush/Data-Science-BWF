# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. How does a Decision Tree algorithm decide where to split?

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X, y)

# Display tree structure
tree_rules = export_text(tree, feature_names=iris['feature_names'])
print("Decision Tree Rules:\n", tree_rules)

# 3. Implement a Decision Tree Classifier in Python using scikit-learn.

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

# Predict on test set
y_pred = tree.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nDecision Tree Accuracy: {accuracy}")

# 4. What is the role of the max_depth parameter in a Decision Tree?

# Train Decision Tree with max_depth
tree_with_depth = DecisionTreeClassifier(max_depth=3, random_state=0)
tree_with_depth.fit(X, y)

# Display tree structure
tree_rules_depth = export_text(tree_with_depth, feature_names=iris['feature_names'])
print("\nDecision Tree with max_depth=3 Rules:\n", tree_rules_depth)

# 5. Implement a Random Forest Classifier in Python using scikit-learn.

# Train Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

# Predict on test set
y_pred_forest = forest.predict(X_test)

# Evaluate model
accuracy_forest = accuracy_score(y_test, y_pred_forest)
print(f"\nRandom Forest Accuracy: {accuracy_forest}")

# 6. How does the n_estimators parameter affect a Random Forest model?

# Train Random Forest with different n_estimators
accuracies = []
for n in [10, 50, 100, 200]:
    forest = RandomForestClassifier(n_estimators=n, random_state=0)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_test)
    accuracies.append((n, accuracy_score(y_test, y_pred)))

# Display accuracies
print("\nEffect of n_estimators on Random Forest Accuracy:")
for n, accuracy in accuracies:
    print(f"n_estimators={n}: Accuracy={accuracy}")

# 7. Explain feature importance in Random Forests.

# Train Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X, y)

# Display feature importances
importances = forest.feature_importances_
feature_names = iris['feature_names']
print("\nFeature Importances in Random Forest:")
for feature, importance in zip(feature_names, importances):
    print(f"{feature}: {importance:.4f}")

# 8. What is the out-of-bag (OOB) error in Random Forests?

# Train Random Forest with OOB score
forest_with_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
forest_with_oob.fit(X, y)

# Display OOB score
print(f"\nOOB Score: {forest_with_oob.oob_score_}")

# 9. How can you visualize a Decision Tree?

# Train Decision Tree
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X, y)

# Visualize tree
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True)
plt.show()
