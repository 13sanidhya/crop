import numpy as np
import pandas as pd
import pickle
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("="*70)
print("ENHANCED CROP RECOMMENDATION MODEL TRAINING")
print("="*70)

# 1. LOAD DATA
print("\n1. Loading Dataset...")
crop = pd.read_csv("Crop_recommendation.csv")
print(f"Dataset shape: {crop.shape}")
print(f"Number of crops: {crop['label'].nunique()}")
print(f"Missing values: {crop.isnull().sum().sum()}")

# 2. ENCODE LABELS
print("\n2. Encoding Labels...")
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop(['label'], axis=1, inplace=True)

# 3. FEATURE ANALYSIS
print("\n3. Feature Statistics:")
print(crop.describe())

# 4. TRAIN-TEST SPLIT (Stratified for balanced classes)
print("\n4. Splitting Data...")
X = crop.drop(['crop_num'], axis=1)
y = crop['crop_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# 5. FEATURE SCALING
print("\n5. Scaling Features...")
ms = MinMaxScaler()
sc = StandardScaler()

X_train_scaled = ms.fit_transform(X_train)
X_test_scaled = ms.transform(X_test)

X_train_final = sc.fit_transform(X_train_scaled)
X_test_final = sc.transform(X_test_scaled)

# 6. BASELINE MODEL COMPARISON
print("\n6. Comparing Multiple Models (with Cross-Validation)...")
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

results = {}
for name, model in models.items():
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_final, y_train, cv=5, scoring='accuracy')
    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'test_accuracy': test_acc,
        'f1_score': f1
    }
    
    print(f"\n{name}:")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")

# 7. HYPERPARAMETER TUNING FOR RANDOM FOREST
print("\n7. Hyperparameter Tuning for Random Forest...")
print("   This may take a few minutes...")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Use RandomizedSearchCV for faster results
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

rf_random.fit(X_train_final, y_train)
best_rf = rf_random.best_estimator_

print(f"\nBest Parameters: {rf_random.best_params_}")
print(f"Best CV Score: {rf_random.best_score_:.4f}")

# 8. EVALUATE BEST MODEL
print("\n8. Final Model Evaluation...")
y_pred_best = best_rf.predict(X_test_final)
final_accuracy = accuracy_score(y_test, y_pred_best)
final_f1 = f1_score(y_test, y_pred_best, average='weighted')

print(f"Final Test Accuracy: {final_accuracy:.4f}")
print(f"Final F1 Score: {final_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# 9. FEATURE IMPORTANCE
print("\n9. Feature Importance Analysis...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance.to_string(index=False))

# 10. ENSEMBLE MODEL (Optional - Voting Classifier)
print("\n10. Creating Ensemble Model...")
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('gb', GradientBoostingClassifier(random_state=42)),
        ('nb', GaussianNB())
    ],
    voting='soft'
)

voting_clf.fit(X_train_final, y_train)
y_pred_ensemble = voting_clf.predict(X_test_final)
ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)

print(f"Ensemble Model Accuracy: {ensemble_accuracy:.4f}")

# 11. SELECT BEST MODEL
print("\n11. Selecting Best Model...")
if ensemble_accuracy > final_accuracy:
    final_model = voting_clf
    final_model_name = "Ensemble (Voting Classifier)"
    final_score = ensemble_accuracy
else:
    final_model = best_rf
    final_model_name = "Tuned Random Forest"
    final_score = final_accuracy

print(f"Selected Model: {final_model_name}")
print(f"Final Accuracy: {final_score:.4f}")

# 12. SAVE MODELS
print("\n12. Saving Models...")
pickle.dump(final_model, open('model.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))

# Save feature importance
feature_importance.to_csv('feature_importance.csv', index=False)

print("\n✅ Models saved successfully!")
print("   - model.pkl")
print("   - standscaler.pkl")
print("   - minmaxscaler.pkl")
print("   - feature_importance.csv")

# 13. MODEL INFORMATION SUMMARY
print("\n" + "="*70)
print("MODEL TRAINING SUMMARY")
print("="*70)
print(f"Best Model: {final_model_name}")
print(f"Accuracy: {final_score:.2%}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"Number of Features: {X.shape[1]}")
print(f"Number of Classes: {y.nunique()}")
print("="*70)

# 14. SAVE MODEL METADATA
metadata = {
    'model_name': final_model_name,
    'accuracy': final_score,
    'f1_score': final_f1,
    'training_samples': len(X_train),
    'testing_samples': len(X_test),
    'features': list(X.columns),
    'best_params': rf_random.best_params_ if final_model_name == "Tuned Random Forest" else None
}

import json
with open('model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)

print("\n✅ Model metadata saved to model_metadata.json")
print("\n🎉 Training Complete! Your improved model is ready to use.")
