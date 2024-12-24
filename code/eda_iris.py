# eda_iris.py

# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset/iris.csv')

# Display the first few rows of the dataset
print("First five rows of the dataset:")
print(data.head())

# Display basic statistics about the dataset
print("\nSummary statistics of the dataset:")
print(data.describe())

# Pairplot for visualizing relationships between features
sns.pairplot(data, hue='species', diag_kind='hist')
plt.title('Pairplot of Iris Features')
plt.show()

# Correlation heatmap
corr_matrix = data.iloc[:, :-2].corr()  # Exclude target and species columns
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap of Iris Features')
plt.show()
