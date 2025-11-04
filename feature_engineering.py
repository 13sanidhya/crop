import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FEATURE ENGINEERING & ANALYSIS")
print("="*70)

# 1. LOAD DATA
print("\n1. Loading dataset...")
crop = pd.read_csv("Crop_recommendation.csv")
print(f"Original shape: {crop.shape}")

# 2. CREATE NEW FEATURES
print("\n2. Creating engineered features...")

# NPK Ratio features
crop['N_P_ratio'] = crop['N'] / (crop['P'] + 1)
crop['N_K_ratio'] = crop['N'] / (crop['K'] + 1)
crop['P_K_ratio'] = crop['P'] / (crop['K'] + 1)

# Total NPK
crop['NPK_sum'] = crop['N'] + crop['P'] + crop['K']
crop['NPK_mean'] = crop['NPK_sum'] / 3

# Climate interaction features
crop['temp_humidity_interaction'] = crop['temperature'] * crop['humidity'] / 100
crop['rainfall_humidity_interaction'] = crop['rainfall'] * crop['humidity'] / 100

# pH categories
crop['pH_category'] = pd.cut(crop['ph'], 
                              bins=[0, 5.5, 6.5, 7.5, 10], 
                              labels=['Acidic', 'Slightly_Acidic', 'Neutral_Alkaline', 'Alkaline'])

# Temperature categories
crop['temp_category'] = pd.cut(crop['temperature'],
                               bins=[0, 15, 25, 35, 50],
                               labels=['Cool', 'Moderate', 'Warm', 'Hot'])

# Rainfall categories
crop['rainfall_category'] = pd.cut(crop['rainfall'],
                                   bins=[0, 100, 200, 300, 400],
                                   labels=['Low', 'Medium', 'High', 'Very_High'])

# Humidity stress indicator
crop['humidity_stress'] = ((crop['humidity'] < 40) | (crop['humidity'] > 90)).astype(int)

# Temperature stress indicator
crop['temp_stress'] = ((crop['temperature'] < 15) | (crop['temperature'] > 35)).astype(int)

# Combined stress
crop['total_stress'] = crop['humidity_stress'] + crop['temp_stress']

# Water requirement index (rainfall/humidity)
crop['water_index'] = crop['rainfall'] / (crop['humidity'] + 1)

# Nutrient balance score
crop['nutrient_balance'] = np.abs(crop['N'] - crop['NPK_mean']) + \
                           np.abs(crop['P'] - crop['NPK_mean']) + \
                           np.abs(crop['K'] - crop['NPK_mean'])

print(f"New shape: {crop.shape}")
print(f"Added {crop.shape[1] - 8} new features")

# 3. CORRELATION ANALYSIS
print("\n3. Analyzing feature correlations...")

# Encode label for correlation
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)

# Select numeric columns
numeric_cols = crop.select_dtypes(include=[np.number]).columns

# Correlation with target
target_corr = crop[numeric_cols].corr()['crop_num'].sort_values(ascending=False)
print("\nTop 10 Features Correlated with Crop Type:")
print(target_corr.head(11).to_string())  # 11 to exclude crop_num itself

# 4. VISUALIZATIONS
print("\n4. Creating visualizations...")

# Feature correlation heatmap
plt.figure(figsize=(16, 14))
corr_matrix = crop[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('static/screenshots/correlation_matrix.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/correlation_matrix.png")

# NPK Distribution by crop
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
crop.boxplot(column='N', by='label', ax=plt.gca(), rot=90)
plt.title('Nitrogen Distribution by Crop')
plt.suptitle('')

plt.subplot(1, 3, 2)
crop.boxplot(column='P', by='label', ax=plt.gca(), rot=90)
plt.title('Phosphorus Distribution by Crop')
plt.suptitle('')

plt.subplot(1, 3, 3)
crop.boxplot(column='K', by='label', ax=plt.gca(), rot=90)
plt.title('Potassium Distribution by Crop')
plt.suptitle('')

plt.tight_layout()
plt.savefig('static/screenshots/npk_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/npk_distribution.png")

# Climate features distribution
plt.figure(figsize=(14, 6))

plt.subplot(1, 3, 1)
sns.histplot(data=crop, x='temperature', hue='temp_category', bins=30, kde=True)
plt.title('Temperature Distribution')

plt.subplot(1, 3, 2)
sns.histplot(data=crop, x='humidity', bins=30, kde=True)
plt.title('Humidity Distribution')

plt.subplot(1, 3, 3)
sns.histplot(data=crop, x='rainfall', bins=30, kde=True)
plt.title('Rainfall Distribution')

plt.tight_layout()
plt.savefig('static/screenshots/climate_distribution.png', dpi=300, bbox_inches='tight')
print("   ✅ Saved: static/screenshots/climate_distribution.png")

# Feature importance of new features
print("\n5. Analyzing new feature importance...")

# Save enhanced dataset
crop_enhanced = crop.copy()
crop_enhanced.drop(['pH_category', 'temp_category', 'rainfall_category'], axis=1, inplace=True)
crop_enhanced.to_csv('Crop_recommendation_enhanced.csv', index=False)
print("\n   ✅ Saved: Crop_recommendation_enhanced.csv")

# 5. FEATURE SUMMARY
print("\n" + "="*70)
print("FEATURE ENGINEERING SUMMARY")
print("="*70)
print(f"Original Features: 7")
print(f"Engineered Features: {crop.shape[1] - 8}")
print(f"Total Features: {crop.shape[1] - 1}")  # Excluding label
print("\nNew Features Created:")
print("  1. NPK Ratios (N/P, N/K, P/K)")
print("  2. NPK Sum and Mean")
print("  3. Climate Interactions (Temp*Humidity, Rainfall*Humidity)")
print("  4. Categorical Features (pH, Temp, Rainfall categories)")
print("  5. Stress Indicators (Humidity, Temperature, Total stress)")
print("  6. Water Index (Rainfall/Humidity)")
print("  7. Nutrient Balance Score")
print("\nTop 5 Most Correlated Features with Crop Type:")
for i, (feature, corr_value) in enumerate(target_corr.head(6).items(), 1):
    if feature != 'crop_num':
        print(f"  {i}. {feature}: {corr_value:.4f}")
print("="*70)

print("\n🎉 Feature Engineering Complete!")
print("   Enhanced dataset ready for improved model training.")
