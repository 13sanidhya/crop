"""
Model Evaluation and Visualization Script
Generate detailed performance metrics and visualizations
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_auc_score, precision_recall_fscore_support)
from sklearn.model_selection import learning_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*70)
print("MODEL EVALUATION & VISUALIZATION")
print("="*70)

# 1. LOAD DATA
print("\n1. Loading data...")
crop = pd.read_csv("Crop_recommendation.csv")

crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}

crop['crop_num'] = crop['label'].map(crop_dict)
X = crop.drop(['label', 'crop_num'], axis=1)
y = crop['crop_num']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. LOAD MODELS
print("2. Loading trained models...")
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Scale data
X_train_scaled = ms.transform(X_train)
X_test_scaled = ms.transform(X_test)
X_train_final = sc.transform(X_train_scaled)
X_test_final = sc.transform(X_test_scaled)

# 3. MAKE PREDICTIONS
print("3. Making predictions...")
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)

# 4. CALCULATE METRICS
print("\n4. Performance Metrics:")
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")

# 5. CONFUSION MATRIX VISUALIZATION
print("\n5. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(16, 14))
crop_names = ['Rice', 'Maize', 'Jute', 'Cotton', 'Coconut', 'Papaya', 'Orange', 'Apple',
              'Muskmelon', 'Watermelon', 'Grapes', 'Mango', 'Banana', 'Pomegranate',
              'Lentil', 'Blackgram', 'Mungbean', 'Mothbeans', 'Pigeonpeas', 'Kidneybeans',
              'Chickpea', 'Coffee']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=crop_names, yticklabels=crop_names)
plt.title('Confusion Matrix - Crop Prediction', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('static/screenshots/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/confusion_matrix.png")

# 6. FEATURE IMPORTANCE
print("\n6. Feature importance analysis...")
if hasattr(model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importance in Crop Recommendation', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/screenshots/feature_importance.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: static/screenshots/feature_importance.png")
    
    print("\n   Feature Ranking:")
    for idx, row in feature_imp.iterrows():
        print(f"   {row['Feature']:15s}: {row['Importance']:.4f}")

# 7. CLASS-WISE PERFORMANCE
print("\n7. Per-class performance:")
class_report = classification_report(y_test, y_pred, target_names=crop_names, output_dict=True, zero_division=0)

class_performance = pd.DataFrame(class_report).transpose()
class_performance = class_performance.iloc[:-3]  # Remove avg rows

plt.figure(figsize=(14, 8))
x = np.arange(len(crop_names))
width = 0.25

plt.bar(x - width, class_performance['precision'], width, label='Precision', alpha=0.8)
plt.bar(x, class_performance['recall'], width, label='Recall', alpha=0.8)
plt.bar(x + width, class_performance['f1-score'], width, label='F1-Score', alpha=0.8)

plt.xlabel('Crops', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
plt.xticks(x, crop_names, rotation=45, ha='right')
plt.legend()
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/screenshots/class_performance.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/class_performance.png")

# 8. PREDICTION CONFIDENCE DISTRIBUTION
print("\n8. Analyzing prediction confidence...")
max_probabilities = np.max(y_pred_proba, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(max_probabilities, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(max_probabilities.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {max_probabilities.mean():.3f}')
plt.xlabel('Prediction Confidence', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Prediction Confidence', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('static/screenshots/confidence_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/confidence_distribution.png")

print(f"\n   Average Confidence: {max_probabilities.mean():.4f}")
print(f"   Min Confidence: {max_probabilities.min():.4f}")
print(f"   Max Confidence: {max_probabilities.max():.4f}")

# 9. LEARNING CURVES
print("\n9. Generating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_final, y_train, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, val_mean, 'o-', color='g', label='Cross-validation Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='g')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.title('Learning Curves', fontsize=16, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('static/screenshots/learning_curves.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/learning_curves.png")

# 10. SUMMARY REPORT
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)
print(f"Total Samples: {len(X_test)}")
print(f"Correct Predictions: {(y_test == y_pred).sum()}")
print(f"Incorrect Predictions: {(y_test != y_pred).sum()}")
print(f"Overall Accuracy: {accuracy:.2%}")
print(f"Average Confidence: {max_probabilities.mean():.2%}")
print("\nTop 5 Most Important Features:")
if hasattr(model, 'feature_importances_'):
    for i, row in feature_imp.head(5).iterrows():
        print(f"  {i+1}. {row['Feature']}: {row['Importance']:.4f}")
print("="*70)

# Save summary to file
with open('model_evaluation_summary.txt', 'w') as f:
    f.write("MODEL EVALUATION SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n")
    f.write(f"Average Confidence: {max_probabilities.mean():.4f}\n")
    f.write("\n" + classification_report(y_test, y_pred, target_names=crop_names, zero_division=0))

print("\n✅ Evaluation summary saved to: model_evaluation_summary.txt")
print("\n🎉 Evaluation Complete! Check the static/screenshots/ folder for visualizations.")
