import seaborn as sns
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Load dataset
titanic_data = sns.load_dataset('titanic')

# Drop all-NaN columns (IterativeImputer fails on them)
titanic_data = titanic_data.dropna(axis=1, how='all')

# Copy and prepare for imputation
titanic_data_filled = titanic_data.copy()

# Identify columns
numerical_cols = titanic_data_filled.select_dtypes(include=[np.number]).columns
categorical_cols = titanic_data_filled.select_dtypes(exclude=[np.number]).columns

# Debug print (optional)
print("Numerical Columns for Imputation:", numerical_cols)

# Impute numerical columns using IterativeImputer
try:
    imputer_num = IterativeImputer(estimator=RandomForestRegressor(), random_state=0)
    titanic_data_filled[numerical_cols] = imputer_num.fit_transform(titanic_data_filled[numerical_cols])
except Exception as e:
    print("❌ IterativeImputer failed:", e)
    print("➡️ Consider switching to SimpleImputer as a fallback.")

# Impute categorical columns using mode
for col in categorical_cols:
    titanic_data_filled[col] = titanic_data_filled[col].fillna(titanic_data_filled[col].mode()[0])

# Export to CSV
titanic_data_filled.to_csv('titanic_cleaned.csv', index=False)
print("✅ CSV file saved as 'titanic_cleaned.csv'")
