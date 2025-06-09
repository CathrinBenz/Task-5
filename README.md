# Task-5 Exploratory Data Analysis
# 🎯 Objective: Titanic Dataset EDA
    The objective of this project is to perform Exploratory Data Analysis (EDA) on the Titanic dataset
    to uncover meaningful insights about passenger survival during the Titanic disaster. By analyzing
    key variables such as age, gender, passenger class, fare, and embarkation port, we aim to:

        Understand the factors that influenced survival.

        Identify missing values and outliers.

        Visualize the relationships between features and the survival rate.

        Prepare the data for potential future modeling (e.g., predictive analysis).

     This analysis helps build foundational understanding of the dataset, which is essential before applying any machine learning or statistical modeling techniques.
# ✅ Dataset Overview:
Rows: 891

Columns: 12

Missing Values:

    Age: 177 missing

    Cabin: 687 missing

    Embarked: 2 missing

Target Column: Survived (0 = No, 1 = Yes)
# ✅ Full EDA Code for Titanic Dataset

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df = pd.read_csv("Titanic-Dataset.csv")  

print("=== HEAD ===")

print(df.head())

print("\n=== INFO ===")

print(df.info())

print("\n=== DESCRIBE ===")

print(df.describe(include='all'))

print("\n=== MISSING VALUES ===")

print(df.isnull().sum())

print("\n=== SURVIVAL COUNTS ===")

print(df['Survived'].value_counts())

df['Age'].fillna(df['Age'].median(), inplace=True)

df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.drop(columns=['Cabin'], inplace=True)

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived')

plt.suptitle("Pairplot: Age, Fare, Survived", y=1.02)

plt.show()

plt.figure(figsize=(8, 5))

sns.boxplot(x='Survived', y='Age', data=df)

plt.title("Boxplot: Age vs Survived")

plt.show()

plt.figure(figsize=(8, 5))

sns.histplot(df['Fare'], kde=True, bins=30)

plt.title("Fare Distribution")

plt.xlabel("Fare")

plt.ylabel("Frequency")

plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(x='Sex', hue='Survived', data=df.replace({0: 'Male', 1: 'Female'}))

plt.title("Survival Count by Sex")

plt.show()

print("\n=== OBSERVATIONS ===")

print("""

1. Women had higher survival rates than men.
 
2. Passengers in 1st class had better chances of survival.
    
3. Younger passengers tended to survive more.
 
4. Fare and survival were positively correlated.

5. Port of Embarkation influenced survival chances.
    
""")
# Screenshots of Output
![Screenshot 2025-06-09 191928](https://github.com/user-attachments/assets/e1c2eec7-80f3-4661-b368-5ac966a120f5)
![Screenshot 2025-06-09 191940](https://github.com/user-attachments/assets/a37f128a-6cc1-44ab-bbb7-01ad6fd83892)
![Screenshot 2025-06-09 191950](https://github.com/user-attachments/assets/651f335f-882e-4ab6-8c25-6aaf5d0a3bda)
![Screenshot 2025-06-09 192008](https://github.com/user-attachments/assets/ff619a74-e8b3-41d2-95aa-a9ac0074f10b)
![Screenshot 2025-06-09 192018](https://github.com/user-attachments/assets/0cd75f51-159c-44ac-a5d0-0b68a7b7685f)
![Screenshot 2025-06-09 192034](https://github.com/user-attachments/assets/da56a4ba-84ad-4a42-bcf8-a5bb1f225ca7)
![Screenshot 2025-06-09 192045](https://github.com/user-attachments/assets/ecb877a9-90e1-4595-a887-5e26cca7c4b5)
![Screenshot 2025-06-09 192057](https://github.com/user-attachments/assets/a1c2e3be-d229-47fc-bbf5-4a3eafac293c)
![Screenshot 2025-06-09 192109](https://github.com/user-attachments/assets/68354280-7364-4c92-af85-05e2b37893a6)
![Screenshot 2025-06-09 192159](https://github.com/user-attachments/assets/06320360-d15f-4280-8767-9ac4a51ddee2)
![Screenshot 2025-06-09 192212](https://github.com/user-attachments/assets/a423b94c-2cb4-42a6-a5f7-b1d2582caa5e)
![Screenshot 2025-06-09 192225](https://github.com/user-attachments/assets/335eaee7-85f8-4a4c-aa2a-7b2cc61f308a)
![Screenshot 2025-06-09 192249](https://github.com/user-attachments/assets/e0191a6b-3558-429b-ba8b-a7d37597f141)
![Screenshot 2025-06-09 192301](https://github.com/user-attachments/assets/ce4024cf-9dca-473e-ac9c-7a1dc8fe1ad7)
![Screenshot 2025-06-09 192334](https://github.com/user-attachments/assets/383b4399-dd5e-47f5-8f28-ead8169ce1a0)





















