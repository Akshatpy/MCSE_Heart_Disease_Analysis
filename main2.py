import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
file_path = 'Heart_disease_statlog.csv'
df = pd.read_csv(file_path)
print("--- Step 0: Artificially Creating Missing Values ---")
print(f"Missing values BEFORE creating them: \n{df.isnull().sum().loc[lambda x: x > 0]}")
np.random.seed(42) #ive put a random seed for introducing missing values
chol_indices = df.sample(n=30).index
df.loc[chol_indices, 'chol'] = np.nan
trestbps_indices = df.sample(n=20).index
df.loc[trestbps_indices, 'trestbps'] = np.nan
print(f"\nMissing values AFTER creating them: \n{df.isnull().sum().loc[lambda x: x > 0]}")
print("\n--- Objective 1: Data Cleaning ---")
# im going to fill those values with median (numeric missing data)
chol_median = df['chol'].median()
df['chol'] = df['chol'].fillna(chol_median)
print(f"Filled 'chol' missing values with median: {chol_median}")
trestbps_median = df['trestbps'].median()
df['trestbps'] = df['trestbps'].fillna(trestbps_median)
print(f"Filled 'trestbps' missing values with median: {trestbps_median}")
print(f"\nMissing values AFTER cleaning: \n{df.isnull().sum().loc[lambda x: x > 0]}")
if df.isnull().sum().sum() == 0:
    print("Data is now clean.")
print("\n--- Objective 2: Descriptive Statistics ---")
print("\nFull Numeric Features Summary:")
print(df.describe().T)
print("\n--- Objective 3: Data Visualization (Saving Plots) ---")
#  Histogram for 'chol' 
plt.figure(figsize=(10, 6))
sns.histplot(df['chol'], kde=True, bins=30)
plt.title('Histogram of Patient Cholesterol (chol)')
plt.xlabel('Cholesterol (mg/dL)')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('chol_histogram.png')
# Box Plot for 'chol' by 'target'
plt.figure(figsize=(12, 7))
sns.boxplot(x='target', y='chol', data=df)
plt.title('Cholesterol Distribution by Heart Disease Target')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Cholesterol (mg/dL)')
plt.savefig('chol_by_target_boxplot.png')
# Scatter Plot for 'age' vs 'thalach'
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='thalach', data=df, hue='target', alpha=0.7)
plt.title('Scatter Plot of Age vs. Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate (thalach)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('age_vs_thalach_scatter.png')
# Correlation Heatmap
plt.figure(figsize=(12, 9))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Heart Disease Features')
plt.savefig('correlation_heatmap.png')
print("Saved correlation_heatmap.png")
print("\n--- Objective 4: Normality Check (Saving Plot) ---")
plt.figure(figsize=(8, 6))
stats.probplot(df['chol'], dist="norm", plot=plt)
plt.title('Q-Q Plot for Cholesterol (chol)')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('chol_qq_plot.png')
# Confidence Interval
print("\n--- Objective 5: 95% Confidence Interval for Mean Cholesterol ---")
data_col = df['chol']
confidence_level = 0.95
degrees_freedom = len(data_col) - 1
mean_val = np.mean(data_col)
std_error_val = stats.sem(data_col) 
ci = stats.t.interval(confidence_level, degrees_freedom, loc=mean_val, scale=std_error_val)
print(f"Mean Cholesterol: {mean_val:,.2f} mg/dL")
print(f"95% Confidence Interval for Mean Cholesterol: (${ci[0]:,.2f}, ${ci[1]:,.2f})")
# Hypothesis Testing
print("\n--- Objective 6: Hypothesis Test for Mean Cholesterol ---")
hypothesized_mean = 200
alpha = 0.05
t_statistic, p_value = stats.ttest_1samp(df['chol'], hypothesized_mean)
print(f"Null Hypothesis (H0): Mean Cholesterol = ${hypothesized_mean}")
print(f"Alternative Hypothesis (Ha): Mean Cholesterol != ${hypothesized_mean}")
print(f"Significance Level (alpha): {alpha}")
print(f"t-statistic: {t_statistic:.2f}")
print(f"p-value: {p_value}")
if p_value < alpha:
    print("Result: Reject the null hypothesis (H0).")
else:
    print("Result: Fail to reject the null hypothesis (H0).")
# LinReg Model
print("\n--- Objective 7: Linear Regression Model ---")
print("Note: Using Linear Regression to predict a binary target (0/1).")
predictors = ['age', 'chol', 'trestbps', 'thalach']
target = 'target'
X = df[predictors]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"Target Variable: '{target}'")
print(f"Predictor Variables: {predictors}")
print("\nModel Performance Metrics:")
print(f"R-squared (R²): {r2:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
# Visualizinng the results (Actual vs. Predicted)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
plt.title('Linear Regression: Actual (0/1) vs. Predicted')
plt.xlabel('Actual Target (0 or 1)')
plt.ylabel('Predicted Target Value')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('regression_actual_vs_pred.png')
