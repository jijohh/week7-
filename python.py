# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")
# Load dataset (using seaborn's iris dataset for demo)
try:
    df = sns.load_dataset("iris")  # Replace with pd.read_csv("dataset.csv") if using your own file
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    print("❌ File not found. Please check the file path.")

# Display first 5 rows
df.head()
# Dataset info
df.info()

# Check missing values
print("Missing values per column:")
print(df.isnull().sum())

# Drop missing values (if any)
df = df.dropna()
## Task 2: Basic Data Analysis
# Descriptive statistics
df.describe()
# Group by species and compute mean
grouped = df.groupby("species").mean()
grouped
# Example finding
print("Average petal length is highest for:", grouped["petal_length"].idxmax())
## Task 3: Data Visualization
---
# Line Chart (cumulative petal length)
df["cumulative_petal_length"] = df["petal_length"].cumsum()

plt.figure(figsize=(8,5))
plt.plot(df.index, df["cumulative_petal_length"], label="Cumulative Petal Length")
plt.title("Line Chart: Cumulative Petal Length Over Index")
plt.xlabel("Index")
plt.ylabel("Cumulative Petal Length")
plt.legend()
plt.show()
# Bar Chart (average petal length per species)
plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal_length", data=df, estimator="mean", errorbar=None)
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.show()
# Histogram (distribution of sepal length)
plt.figure(figsize=(8,5))
plt.hist(df["sepal_length"], bins=15, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Length Distribution")
plt.xlabel("Sepal Length")
plt.ylabel("Frequency")
plt.show()
# Scatter Plot (sepal length vs petal length)
plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df, s=80)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend(title="Species")
plt.show()
## Observations & Findings
- The dataset has 3 species of Iris flowers.
- Petal length and petal width are strong indicators for distinguishing species.
- Average petal length is highest for **Iris-virginica**.
- Sepal length distribution is roughly normal but varies across species.
