"""
Train and Evaluate Heart Disease Prediction Models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data
df = pd.read_csv('heart_disease_clean.csv')

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("=" * 60)
print("DATA SPLIT")
print("=" * 60)
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_train_proba = lr_model.predict_proba(X_train_scaled)[:, 1]
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)
lr_roc_auc = roc_auc_score(y_test, lr_test_proba)

print(f"Training Accuracy: {lr_train_acc:.4f}")
print(f"Test Accuracy: {lr_test_acc:.4f}")
print(f"ROC AUC Score: {lr_roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, lr_test_pred)}")

# Train Decision Tree
print("\n" + "=" * 60)
print("DECISION TREE")
print("=" * 60)
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)

dt_train_pred = dt_model.predict(X_train)
dt_test_pred = dt_model.predict(X_test)
dt_train_proba = dt_model.predict_proba(X_train)[:, 1]
dt_test_proba = dt_model.predict_proba(X_test)[:, 1]

dt_train_acc = accuracy_score(y_train, dt_train_pred)
dt_test_acc = accuracy_score(y_test, dt_test_pred)
dt_roc_auc = roc_auc_score(y_test, dt_test_proba)

print(f"Training Accuracy: {dt_train_acc:.4f}")
print(f"Test Accuracy: {dt_test_acc:.4f}")
print(f"ROC AUC Score: {dt_roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, dt_test_pred)}")

# Feature importance
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

# Logistic Regression coefficients
lr_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Coefficient', ascending=False)
print("\nLogistic Regression - Top Features:")
print(lr_importance.head(10))

# Decision Tree feature importance
dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\nDecision Tree - Top Features:")
print(dt_importance.head(10))

# Create visualizations
fig = plt.figure(figsize=(20, 12))

# 1. Confusion Matrix - Logistic Regression
plt.subplot(2, 4, 1)
cm_lr = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Logistic Regression\nConfusion Matrix', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 2. Confusion Matrix - Decision Tree
plt.subplot(2, 4, 2)
cm_dt = confusion_matrix(y_test, dt_test_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False)
plt.title('Decision Tree\nConfusion Matrix', fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 3. ROC Curve
plt.subplot(2, 4, 3)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_test_proba)
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_test_proba)
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC={lr_roc_auc:.3f})', linewidth=2)
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC={dt_roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# 4. Model Comparison
plt.subplot(2, 4, 4)
models = ['Logistic\nRegression', 'Decision\nTree']
train_accs = [lr_train_acc, dt_train_acc]
test_accs = [lr_test_acc, dt_test_acc]
x = np.arange(len(models))
width = 0.35
plt.bar(x - width/2, train_accs, width, label='Train', color='skyblue')
plt.bar(x + width/2, test_accs, width, label='Test', color='orange')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison', fontweight='bold')
plt.xticks(x, models)
plt.legend()
plt.ylim([0.7, 1.0])
for i, v in enumerate(train_accs):
    plt.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(test_accs):
    plt.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

# 5. Feature Importance - Logistic Regression
plt.subplot(2, 4, 5)
top_lr = lr_importance.head(10)
plt.barh(range(len(top_lr)), top_lr['Coefficient'], color='steelblue')
plt.yticks(range(len(top_lr)), top_lr['Feature'])
plt.xlabel('Absolute Coefficient')
plt.title('Logistic Regression\nTop 10 Features', fontweight='bold')
plt.gca().invert_yaxis()

# 6. Feature Importance - Decision Tree
plt.subplot(2, 4, 6)
top_dt = dt_importance.head(10)
plt.barh(range(len(top_dt)), top_dt['Importance'], color='forestgreen')
plt.yticks(range(len(top_dt)), top_dt['Feature'])
plt.xlabel('Importance')
plt.title('Decision Tree\nTop 10 Features', fontweight='bold')
plt.gca().invert_yaxis()

# 7. Prediction Distribution - LR
plt.subplot(2, 4, 7)
plt.hist([lr_test_proba[y_test==0], lr_test_proba[y_test==1]],
         bins=20, label=['No Disease', 'Disease'], color=['green', 'red'], alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Logistic Regression\nPrediction Distribution', fontweight='bold')
plt.legend()

# 8. Prediction Distribution - DT
plt.subplot(2, 4, 8)
plt.hist([dt_test_proba[y_test==0], dt_test_proba[y_test==1]],
         bins=20, label=['No Disease', 'Disease'], color=['green', 'red'], alpha=0.7)
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Decision Tree\nPrediction Distribution', fontweight='bold')
plt.legend()

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 60)
print("Model evaluation visualizations saved to 'model_evaluation.png'")
print("=" * 60)
