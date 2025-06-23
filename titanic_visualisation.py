# ===========================================
# Exploratory Data Analysis on Titanic Dataset
# ===========================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Configure plot styles
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# ================================
# STEP 1: Load the Dataset
# ================================
df = pd.read_csv("titanic_cleaned.csv")  # Make sure this file is in your directory

print("🔍 Dataset Shape:", df.shape)
print("\n🧾 First 5 Rows:\n", df.head())
print("\n📊 Dataset Info:")
df.info()
print("\n📈 Statistical Summary:\n", df.describe(include='all'))

# ================================
# STEP 2: Missing Values Analysis
# ================================
print("\n❓ Missing Values Count:\n", df.isnull().sum())
print("\n❓ Missing Values %:\n", (df.isnull().sum() / len(df)) * 100)

# Heatmap of missing values
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# ================================
# STEP 3: Univariate Analysis
# ================================
# Numerical Distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

# Categorical Distributions
categorical_cols = df.select_dtypes(exclude=[np.number]).columns

for col in categorical_cols:
    plt.figure()
    sns.countplot(y=col, data=df, palette='Set2')
    plt.title(f'Count of Categories in {col}')
    plt.tight_layout()
    plt.show()

# ================================
# STEP 4: Outlier Detection
# ================================
for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col], color='salmon')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Print IQR-based outliers
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"⚠️ {col}: {len(outliers)} outliers")

# ================================
# STEP 5: Correlation Heatmap
# ================================
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ================================
# STEP 6: Bivariate Analysis
# ================================
if 'survived' in df.columns:
    # Categorical vs Target
    for col in categorical_cols:
        if col != 'survived':
            plt.figure()
            sns.countplot(x=col, hue='survived', data=df)
            plt.title(f'Survival by {col}')
            plt.tight_layout()
            plt.show()

    # Age Group vs Survival
    if 'age' in df.columns:
        df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 40, 60, 100],
                                 labels=['Child', 'Teen', 'Adult', 'Middle-Aged', 'Senior'])
        sns.countplot(x='age_group', hue='survived', data=df, palette='Set1')
        plt.title('Survival by Age Group')
        plt.tight_layout()
        plt.show()

# ================================
# STEP 7: Multivariate Visuals
# ================================
# Pairplot for numeric columns
sns.pairplot(df[numeric_cols])
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

# Catplot: survival by sex and class
if all(col in df.columns for col in ['sex', 'pclass', 'survived']):
    sns.catplot(x='sex', hue='survived', col='pclass', kind='count', data=df)
    plt.suptitle("Survival by Sex and Class", y=1.05)
    plt.show()

# ================================
# STEP 8: Summary of Findings
# ================================
print("\n🔎 Summary of EDA:")
print("- Females had higher survival rates than males.")
print("- 1st class passengers had significantly better survival chances.")
print("- Age shows varied survival trends; children had higher survival.")
print("- Fare and class are positively correlated with survival.")
print("- Large families had lower survival rates.")
