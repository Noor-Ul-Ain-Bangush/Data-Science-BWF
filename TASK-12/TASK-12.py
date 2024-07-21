#                          <<<------EXAMPLE-01------->>>
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# Sample true labels and predictions for binary classification
y_true_binary = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_pred_binary = np.array([1, 0, 1, 0, 0, 1, 0, 1, 1, 0])

# Calculate confusion matrix
cm_binary = confusion_matrix(y_true_binary, y_pred_binary)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_true_binary, y_pred_binary)
precision = precision_score(y_true_binary, y_pred_binary)
recall = recall_score(y_true_binary, y_pred_binary)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')

# Plot confusion matrix as heatmap
sns.heatmap(cm_binary, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


#          <<<-------Example-02: F1 Score----------->>>
from sklearn.metrics import f1_score

# Calculate F1 Score
f1 = f1_score(y_true_binary, y_pred_binary)
print(f'F1 Score: {f1}')


#         <<<--------Example-03: ROC Curve and AUC--------->>>>
from sklearn.metrics import roc_curve, roc_auc_score

# Generate ROC curve values
fpr, tpr, thresholds = roc_curve(y_true_binary, y_pred_binary)

# Calculate AUC
auc = roc_auc_score(y_true_binary, y_pred_binary)
print(f'AUC: {auc}')

# Plot ROC curve
plt.plot(fpr, tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#      <<<--------Example-04: Cross-Validation----------->>>
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Use the Iris dataset for cross-validation
from sklearn.datasets import load_iris
data = load_iris()
X = data.data
y = data.target

# Example classifier
model = RandomForestClassifier()

# Perform cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f'Cross-Validation Accuracy Scores: {cv_scores}')
print(f'Mean Cross-Validation Accuracy: {cv_scores.mean()}')


#     <<<--------Example-05: Precision-Recall Curve------------->>>
from sklearn.metrics import precision_recall_curve

# Generate precision-recall curve values
precision, recall, thresholds = precision_recall_curve(y_true_binary, y_pred_binary)

# Plot Precision-Recall curve
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


#          <<<---------Example-06: Confusion Matrix with More Classes------------>>>
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred_multi = model.predict(X_test)

# Compute confusion matrix
cm_multi = confusion_matrix(y_test, y_pred_multi)

# Plot confusion matrix as heatmap
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Multi-Class)')
plt.show()


#     <<<-----------Example-07: Classification Report---------->>>>
from sklearn.metrics import classification_report

# Generate classification report for binary classification
report_binary = classification_report(y_true_binary, y_pred_binary, target_names=['Class 0', 'Class 1'])
print(report_binary)

# Generate classification report for multi-class classification
report_multi = classification_report(y_test, y_pred_multi, target_names=data.target_names)
print(report_multi)
