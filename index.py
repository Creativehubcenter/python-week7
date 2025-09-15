import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    df = sns.load_dataset("iris")
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print("f Unexpected error: {e}")

print("\nFirst 5 rows of dataset:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values per column:")
print(df.isnull().sum())

df = df.dropna()

print("\nBasic Statistics:")
print(df.describe())

group_means = df.groupby("species").mean()
print("\nMean values grouped by Species:")
print(group_means)

plt.figure(figsize=(8,5))
plt.plot(df.index, df["sepal_length"], label="Sepal Length")
plt.title("Line Chart: Sepal Length Trend")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x="species", y="petal_length", data=df, ci=None)
plt.title("Bar Chart: Avg Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(df["sepal_width"], bins=20, color="skyblue", edgecolor="black")
plt.title("Histogram: Sepal Width Distribution")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df)
plt.title("Scatter Plot: Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()

print("\n Observations:")
print("- Setosa species have much smaller petal length compared to Versicolor and Virginica.")
print("- Versicolor and Virginica overlap in measurements, making them less separable.")
print("- Sepal width is roughly bell-shaped but slightly skewed.")
print("- Scatter plot shows Setosa is distinct, while Versicolor and Virginica are more similar.")
