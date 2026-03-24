"""
Exploratory Data Analysis for Heart Disease Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv('heart_disease_clean.csv')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# 1. Target distribution
plt.subplot(3, 4, 1)
df['target'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Target Distribution\n(0=No Disease, 1=Disease)', fontsize=12, fontweight='bold')
plt.xlabel('Target')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 2. Age distribution by target
plt.subplot(3, 4, 2)
df.boxplot(column='age', by='target', ax=plt.gca())
plt.title('Age Distribution by Target', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Target')
plt.ylabel('Age')

# 3. Sex distribution
plt.subplot(3, 4, 3)
sex_target = pd.crosstab(df['sex'], df['target'])
sex_target.plot(kind='bar', ax=plt.gca(), color=['green', 'red'])
plt.title('Sex vs Target\n(0=Female, 1=Male)', fontsize=12, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease'])

# 4. Chest pain type distribution
plt.subplot(3, 4, 4)
cp_target = pd.crosstab(df['cp'], df['target'])
cp_target.plot(kind='bar', ax=plt.gca(), color=['green', 'red'])
plt.title('Chest Pain Type vs Target', fontsize=12, fontweight='bold')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease'])

# 5. Cholesterol distribution
plt.subplot(3, 4, 5)
df.boxplot(column='chol', by='target', ax=plt.gca())
plt.title('Cholesterol by Target', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Target')
plt.ylabel('Cholesterol')

# 6. Max heart rate distribution
plt.subplot(3, 4, 6)
df.boxplot(column='thalach', by='target', ax=plt.gca())
plt.title('Max Heart Rate by Target', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Target')
plt.ylabel('Max Heart Rate')

# 7. Resting blood pressure
plt.subplot(3, 4, 7)
df.boxplot(column='trestbps', by='target', ax=plt.gca())
plt.title('Resting BP by Target', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Target')
plt.ylabel('Resting BP')

# 8. Exercise induced angina
plt.subplot(3, 4, 8)
exang_target = pd.crosstab(df['exang'], df['target'])
exang_target.plot(kind='bar', ax=plt.gca(), color=['green', 'red'])
plt.title('Exercise Angina vs Target', fontsize=12, fontweight='bold')
plt.xlabel('Exercise Angina')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease'])

# 9. Correlation heatmap
plt.subplot(3, 4, 9)
corr_matrix = df.corr()
sns.heatmap(corr_matrix[['target']].sort_values(by='target', ascending=False),
            annot=True, cmap='coolwarm', center=0, ax=plt.gca(), cbar=False)
plt.title('Feature Correlation with Target', fontsize=12, fontweight='bold')

# 10. Age histogram
plt.subplot(3, 4, 10)
plt.hist([df[df['target']==0]['age'], df[df['target']==1]['age']],
         bins=15, label=['No Disease', 'Disease'], color=['green', 'red'], alpha=0.7)
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

# 11. Oldpeak distribution
plt.subplot(3, 4, 11)
df.boxplot(column='oldpeak', by='target', ax=plt.gca())
plt.title('ST Depression (Oldpeak) by Target', fontsize=12, fontweight='bold')
plt.suptitle('')
plt.xlabel('Target')
plt.ylabel('Oldpeak')

# 12. Number of major vessels
plt.subplot(3, 4, 12)
ca_target = pd.crosstab(df['ca'], df['target'])
ca_target.plot(kind='bar', ax=plt.gca(), color=['green', 'red'])
plt.title('Number of Major Vessels vs Target', fontsize=12, fontweight='bold')
plt.xlabel('Number of Vessels')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['No Disease', 'Disease'])

plt.tight_layout()
plt.savefig('eda_visualizations.png', dpi=300, bbox_inches='tight')
print("EDA visualizations saved to 'eda_visualizations.png'")

# Print correlation analysis
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)
target_corr = df.corr()['target'].sort_values(ascending=False)
print("\nFeatures correlation with target:")
print(target_corr)

# Print summary statistics by target
print("\n" + "=" * 60)
print("SUMMARY STATISTICS BY TARGET")
print("=" * 60)
print("\nNo Disease (0):")
print(df[df['target']==0].describe())
print("\nDisease (1):")
print(df[df['target']==1].describe())
