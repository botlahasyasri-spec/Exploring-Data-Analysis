# ==========================================================
# EXPLORATORY DATA ANALYSIS
# Teen Mental Health Dataset (Automatic Version)
# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Teen_Mental_Health_Dataset.csv")

# -----------------------------
# Dataset Overview
# -----------------------------
print("="*60)
print("FIRST 5 ROWS")
print(df.head())

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nInfo")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary")
print(df.describe(include='all'))

# -----------------------------
# Data Cleaning
# -----------------------------
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nRemaining Missing Values:", df.isnull().sum().sum())

# -----------------------------
# Function to find columns
# -----------------------------
def find_column(keywords):
    cols = [c.lower() for c in df.columns]
    for key in keywords:
        for i, col in enumerate(cols):
            if key in col:
                return df.columns[i]
    return None

age = find_column(["age"])
gender = find_column(["gender", "sex"])
platform = find_column(["platform", "social"])
stress = find_column(["stress"])
depression = find_column(["depression"])
screen = find_column(["screen"])
sleep = find_column(["sleep"])
anxiety = find_column(["anxiety"])
addiction = find_column(["addiction"])
academic = find_column(["academic", "score"])

# -----------------------------
# Visualizations
# -----------------------------
if age:
    plt.figure(figsize=(8,5))
    sns.histplot(df[age], bins=10, kde=True)
    plt.title("Age Distribution")
    plt.show()

if gender:
    plt.figure(figsize=(6,5))
    sns.countplot(x=gender, data=df)
    plt.title("Gender Distribution")
    plt.xticks(rotation=45)
    plt.show()

if platform:
    plt.figure(figsize=(8,5))
    sns.countplot(x=platform, data=df)
    plt.title("Platform Usage")
    plt.xticks(rotation=45)
    plt.show()

if stress:
    plt.figure(figsize=(8,5))
    sns.countplot(x=stress, data=df)
    plt.title("Stress Level")
    plt.xticks(rotation=45)
    plt.show()

if depression:
    plt.figure(figsize=(6,5))
    sns.countplot(x=depression, data=df)
    plt.title("Depression")
    plt.xticks(rotation=45)
    plt.show()

if screen:
    plt.figure(figsize=(8,5))
    sns.histplot(df[screen], kde=True)
    plt.title("Screen Time")
    plt.show()

if sleep:
    plt.figure(figsize=(8,5))
    sns.histplot(df[sleep], kde=True)
    plt.title("Sleep Hours")
    plt.show()

# -----------------------------
# Boxplots
# -----------------------------
if screen:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[screen])
    plt.title("Screen Time Boxplot")
    plt.show()

if sleep:
    plt.figure(figsize=(8,5))
    sns.boxplot(x=df[sleep])
    plt.title("Sleep Hours Boxplot")
    plt.show()

# -----------------------------
# Correlation Heatmap
# -----------------------------
numeric_df = df.select_dtypes(include=np.number)

if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

# -----------------------------
# Pairplot
# -----------------------------
pair_cols = [c for c in [screen, sleep, academic, stress, anxiety, addiction] if c]

if len(pair_cols) >= 2:
    sns.pairplot(df[pair_cols])
    plt.show()

# -----------------------------
# Key Factors
# -----------------------------
if stress:
    print("\nStress Levels")
    print(df[stress].value_counts())

if gender:
    print("\nGender")
    print(df[gender].value_counts())

if depression:
    print("\nDepression")
    print(df[depression].value_counts())

if screen:
    print("\nAverage Screen Time:", df[screen].mean())

if sleep:
    print("\nAverage Sleep Hours:", df[sleep].mean())

if anxiety:
    print("\nAverage Anxiety:", df[anxiety].mean())

if addiction:
    print("\nAverage Addiction:", df[addiction].mean())

print("\n========== EDA COMPLETED ==========")
