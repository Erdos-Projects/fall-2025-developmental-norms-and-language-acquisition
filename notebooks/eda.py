# %% [markdown]
# ---
# title: Initial Exploration and Analysis
# date: 2025-05-15
# ---

# %% [markdown]
#
# ## Load the package and data


# %%

# Packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the training data with features
train_dat = pd.read_parquet("data/processed/train_slam_with_features.parquet")

# %% [markdown]
# ## Basic summary statistics

# %%
train_dat.info()
train_dat.describe()

# %%
# Number of unique users
num_unique_users = train_dat['user'].nunique()
print(f"Number of unique users: {num_unique_users}")

# Number of unique users by l1-l2 combination
user_lang_stats = train_dat.groupby(['l1', 'l2'])['user'].nunique().reset_index()
print("Number of unique users by L1-L2 combination:")
print(user_lang_stats)

# Number of unique tokens
num_unique_tokens = train_dat['token'].nunique()
print(f"Number of unique tokens: {num_unique_tokens}")

# Number of unique tokens by l1-l2 combination
token_lang_stats = train_dat.groupby(['l1', 'l2'])['token'].nunique().reset_index()
print("Number of unique tokens by L1-L2 combination:")
print(token_lang_stats)

# Plot scatterplot of the percentage token_wrong versus median_aoa by token with line of best fit
token_stats = train_dat.groupby('token').agg(
    token_wrong_rate=('token_wrong', 'mean'),
    median_aoa=('median_aoa', 'first'),
    growth_rate=('growth_rate', 'first'),
    l1=('l1', 'first'),
    l2=('l2', 'first')
).reset_index()
plt.figure(figsize=(10, 6))
sns.scatterplot(x = 'median_aoa', y = 'token_wrong_rate', hue='l2', alpha=0.5, data=token_stats)
plt.title('Token Wrong Rate vs Median AoA')
plt.xlabel('Median Age of Acquisition')
plt.ylabel('Token Wrong Rate')
plt.show()

# Plot scatterplot of the percentage token_wrong versus growth_rate by token with line of best fit
plt.figure(figsize=(10, 6))
sns.scatterplot(x = 'growth_rate', y = 'token_wrong_rate', hue='l2', alpha=0.5, data=token_stats)
plt.title('Token Wrong Rate vs Growth Rate AoA')
plt.xlabel('Growth Rate of Age of Acquisition')
plt.ylabel('Token Wrong Rate')
plt.show()

train_dat['session'].value_counts()
train_dat['format'].value_counts()

# Count number of rows per user
user_stats = train_dat.groupby('user').agg(
    num_rows=('token_id', 'size'),
    num_correct=('token_wrong', 'sum'),
    accuracy=('token_wrong', 'mean'),
    max_days=('days', 'max'),
    median_time=('time', 'median'),
    num_unique_tokens=('token', 'nunique')
).reset_index()

# Plot histogram of number of rows per user in Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['num_rows'], bins=30, kde=False)
plt.title('Distribution of Number of Rows per User')
plt.xlabel('Number of Rows')
plt.ylabel('Count of Users')
plt.show()

# Plot accuracy distribution in Seaborn
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['accuracy'], bins=30, kde=False)
plt.title('Distribution of User Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Count of Users')
plt.show()

# Plot of the max days per user
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['max_days'], kde=False)
plt.title('Distribution of Max Days per User')
plt.xlabel('Max Days')
plt.ylabel('Count of Users')
plt.show()

# Plot of the median per user
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['median_time'], kde=False)
plt.title('Distribution of median time per User')
plt.xlabel('Median time')
plt.ylabel('Count of Users')
plt.show()

# Plot number of unique tokens per user
plt.figure(figsize=(10, 6))
sns.histplot(user_stats['num_unique_tokens'], bins=30, kde=False)
plt.title('Distribution of Number of Unique Tokens per User')
plt.xlabel('Number of Unique Tokens')
plt.ylabel('Count of Users')
plt.show()
