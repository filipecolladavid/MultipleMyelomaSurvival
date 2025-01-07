# %%
import sys
import os

# Append the parent directory to the Python path
#sys.path.append(os.path.abspath(".."))

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add the parent directory to the Python path if not already added
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.model_selection import train_test_split
from scripts.gradient_descent import CustomLinearRegression


from scripts.utils import (
    load_kaggle_df, 
    load_starting_df, 
    create_submission_csv, 
    X_cols_universal, 
    y_cols_universal, 
    save_model
)

from scripts.models import ( 
    create_baseline_model,
)
from scripts.evaluation import (
    cMSE_error,
    gradient_cMSE_error, 
    compare_models
)
from scripts.plots import plot_y_yhat

startingDF = load_starting_df()
kaggleDF = load_kaggle_df()

# %% [markdown]
# ## Task 1 - Setting the baseline

# %% [markdown]
# Just like we did in the previous assignment, our first step will be to set a baseline. This task envolves the use of regression and we will fit as a baseline a Linear Regression model that we can use to compare with further model developments.

# %% [markdown]
# ### Task 1.1 - Data preparation and Validation pipeline

# %%
startingDF.head()

# %%
startingDF.info()

# %%
startingDF.describe()

# %% [markdown]
# From a first inspection, we can observe that we have 400 patients, some with missing that. We start by renaming the first column as ID

# %%
startingDF = startingDF.rename(columns={"Unnamed: 0":"ID"})

# %% [markdown]
# Bar plots of the missing values

# %%
msno.bar(startingDF)

# %% [markdown]
# By the first analysis of this plot, we can conclude that:
# - "Genetic Risk" has 85 missing values (400-315).
# - "Comorbity Index" has 45 missing values (400-355)
# - "Treatment Response" has 29 missing values (400 - 371)
# - "Survavil Time" has 160 missing values (400 - 240)

# %% [markdown]
# Plot of the heat map

# %%
msno.heatmap(startingDF)

# %%
msno.matrix(startingDF)

# %% [markdown]
# #### Investigating the "Censored" entries

# %%
numCensored = startingDF[startingDF["Censored"] == 1].shape[0]
numNotCensored = startingDF[startingDF["Censored"] == 0].shape[0]
print("Number of censored patients: ", numCensored)
print("Number of not censored patients: ", numNotCensored)

plt.figure(figsize=(8, 6))
sns.barplot(x=['Censored', 'Not Censored'], y=[numCensored, numNotCensored])
plt.title('Number of Censored vs Not Censored Patients')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# We start by investigating the NaN entries

# %%
DFofCensored = startingDF[startingDF["Censored"] == 1]
DFofCensored = DFofCensored.drop(columns=["Censored"])
DFofCensored[DFofCensored.isnull().any(axis=1)].head(20)

# %%
DFofCensored[DFofCensored["GeneticRisk"].isnull()] #Age is very similar in the case of NaN values of GeneticRisk
                                                   #What about the other way around?
                                                   
plt.figure(figsize=(8, 6))
sns.boxplot(x=DFofCensored["GeneticRisk"].isnull(), y=DFofCensored["Age"])
plt.title("Age Distribution for Patients with and without NaN Values in GeneticRisk")
plt.xlabel("GeneticRisk is NaN")
plt.ylabel("Age")
plt.show()

# %%
DFofCensored[DFofCensored["GeneticRisk"].isnull()].describe() # Maximum age to make the GeneticRisk test is below 70 

# %%
DFofCensored[DFofCensored["GeneticRisk"].notnull()].describe() # Maximum age to make the GeneticRisk test is below 70

# %%
DFofCensored[DFofCensored["ComorbidityIndex"].isnull()]
plt.figure(figsize=(8, 6))
sns.boxplot(x=DFofCensored["ComorbidityIndex"].isnull(), y=DFofCensored["Age"])
plt.title("Age Distribution for Patients with and without NaN Values in ComorbidityIndex")
plt.xlabel("ComorbidityIndex is NaN")
plt.ylabel("Age")
plt.show()

# %%
DFofCensored[DFofCensored["TreatmentResponse"].isnull()]
plt.figure(figsize=(8, 6))
sns.boxplot(x=DFofCensored["TreatmentResponse"].isnull(), y=DFofCensored["Age"])
plt.title("Age Distribution for Patients with and without NaN Values in TreatmentResponse")
plt.xlabel("TreatmentResponse is NaN")
plt.ylabel("Age")
plt.show()

# %% [markdown]
# #### Point 1: If one drops all the data points with missing values, plus the censored ones, would it be possible to fit a model?

# %%
#Drop NaN values
DFnoNA = startingDF.dropna()
#Confirm that there are no NaN values
DFnoNA.isnull().any()

# %%
print("Initial number of rows: ", startingDF.shape[0])
print("Number of current rows: ", DFnoNA.shape[0])
print("Number of rows dropped: ", startingDF.shape[0] - DFnoNA.shape[0])
plt.figure(figsize=(8, 6))
sns.barplot(x=['Initial', 'NaN Dropped'], y=[startingDF.shape[0], DFnoNA.shape[0]])
plt.title('Data Points: Initial vs NaN Dropped')
plt.ylabel('Count')
plt.show()

# %%
#Drop censored data - We should change the name to a different one to avoid different datasets with same name
DFnoNACensored = DFnoNA[DFnoNA["Censored"] != 1]
#Confirm that there are no censored values
DFnoNACensored["Censored"].unique()

# %%
#Print the number of starting rows and the number of rows after dropping censored data
print("Initial number of rows: ", startingDF.shape[0])
print("Number of current rows (after droping NaN and censored entries): ", DFnoNACensored.shape[0])
print("Number of rows dropped: ", startingDF.shape[0] - DFnoNACensored.shape[0])
plt.figure(figsize=(8, 6))
sns.barplot(x=['Initial', 'Dropped Censored and NaN'], y=[startingDF.shape[0], DFnoNACensored.shape[0]])
plt.title('Data Points: Initial vs Dropped Censored and NaN')
plt.ylabel('Count')
plt.show()

# %%
X = DFnoNACensored[X_cols_universal]
y = DFnoNACensored[y_cols_universal]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)

print("Number of patients in training set: ", X_train.shape[0])
print("Number of patients in test set: ", X_test.shape[0])
print("Number of patients in validation set: ", X_val.shape[0])

plt.figure(figsize=(8, 6))
sns.barplot(x=['Training Set', 'Test Set', 'Validation Set'], y=[X_train.shape[0], X_test.shape[0], X_val.shape[0]])
plt.title('Number of Patients in Each Set')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# After cleaning the dataset, we end up with only 109 patients. While it’s technically possible to proceed with an 80/10/10 split and use data from 88 patients for training, this might not be the most effective approach. Given the limited number of patients provided (400), we can and, most likely, should explore methods to "fill" these gaps instead of dropping patients with missing information. This way, we can maximize the available data and retain more patients for modeling.

# %% [markdown]
# All in all, although continuing with this subset is possible, a different approach that preserves more data would likely yield better insights.

# %% [markdown]
# #### Point 2: Drop the columns containing the features with missing data and the censored data points, or missing survival time. How many points are there left?

# %%
missing_columns = startingDF.columns[startingDF.isnull().any()]
missing_columns = missing_columns[missing_columns != 'SurvivalTime'] #We shouldn't remove the label column
print(f"Columns with missing data (excluding 'SurvivalTime'): {missing_columns.tolist()}")

# %%
startingDF_cleaned = startingDF.drop(columns=missing_columns)
startingDF_cleaned = startingDF_cleaned[(startingDF_cleaned['Censored'] == 0) & (startingDF_cleaned['SurvivalTime'].notna())]
startingDF_cleaned = startingDF_cleaned.drop(columns=['Censored'])

# %%
#Print the remaining columns
print(startingDF_cleaned.columns.tolist()) # Remaining columns after dropping columns with missing data and censored data

# %%
remaining_points = startingDF_cleaned.shape[0]
print(f"Initial number of data points: {startingDF.shape[0]}")
print(f"Remaining data points: {remaining_points}")
plt.figure(figsize=(8, 6))
sns.barplot(x=['Initial', 'Columns Drop + Censored'], y=[startingDF.shape[0], remaining_points])
plt.title('Data Points: Initial vs Columns + Censored Drop')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# After removing the features with missing values — namely "GeneticRisk", "ComorbidityIndex", and "TreatmentResponse"— we were left with the following columns: "ID", "Age", "Gender", "Stage", "TreatmentType", and "SurvivalTime". It is worth noting that the "ID" column has little predictive value and can arguably be dropped.    
# Following the removal of entries with missing or censored data, the dataset, originally consisting of 400 rows, was reduced to 161 rows, resulting in a loss of nearly almost 60% (59.75%) of the data.

# %%
X_1 = startingDF_cleaned[["Age", "Gender", "Stage", "TreatmentType"]]
y_1 = startingDF_cleaned[["Age", "Gender", "Stage", "TreatmentType"]]

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_1, y_1, test_size=0.1)
X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train_1, y_train_1, test_size=0.1)

print("Number of patients in training set: ", X_train.shape[0])
print("Number of patients in test set: ", X_test.shape[0])
print("Number of patients in validation set: ", X_val.shape[0])

plt.figure(figsize=(8, 6))
sns.barplot(x=['Training Set', 'Test Set', 'Validation Set'], y=[X_train.shape[0], X_test.shape[0], X_val.shape[0]])
plt.title('Number of Patients in Each Set')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# It's worth mention that, on one hand, removing the features with NaN values in addition to Censored data points allows us to retain more data points and, therefore, more information overall. However, a potential drawback is that this approach removes crucial information that is likely valuable for the models we aim to create.

# %%
print(f"Initial number of data points: {startingDF.shape[0]}")
print(f"Remaining data points after dropping columns: {remaining_points}")
print("Remaining data points after droping NaN and censored entries: ", DFnoNA.shape[0])

# %%
plt.figure(figsize=(8, 6))
sns.barplot(x=['Initial', 'NaN+Censored', 'Columns with NaN + Censored'], y=[startingDF.shape[0], DFnoNA.shape[0], remaining_points])
plt.title('Comparing the Dropping Approaches')
plt.ylabel('Count')
plt.show()

# %% [markdown]
# #### Point 3: Check the pairplot between the remaining features and the target variable. Analyze and comment in the slides.

# %%
# Identify columns without missing data
remaining_columns = startingDF_cleaned.columns
remaining_columns = remaining_columns.drop('ID') #We don't want to use the ID column as a feature
remaining_columns

# %%
pairplot_data = startingDF_cleaned[remaining_columns]
sns.pairplot(pairplot_data)
plt.suptitle("Pairplot of Remaining Features and SurvivalTime", y=1.02)
plt.show()

sns.pairplot(pairplot_data, x_vars=['Age', 'Gender', 'Stage', 'TreatmentType'], y_vars='SurvivalTime')
plt.suptitle("Pairplot of Remaining Features and SurvivalTime", y=1.02)
plt.show()

# %%
corr = startingDF[remaining_columns].corr()
sns.heatmap(corr,annot=True)

# %%
plt.figure(figsize=(6, 4))
sns.scatterplot(x='Age', y='SurvivalTime', data=startingDF)
plt.title("Scatter Plot: Age vs SurvivalTime")
plt.show()

correlation_age_survival = startingDF[['Age', 'SurvivalTime']].corr().iloc[0, 1]
print(f"Correlation between Age and Surviva lTime: {correlation_age_survival:.2f}")


# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.boxplot(x='Gender', y='SurvivalTime', data=startingDF, ax=axes[0])
axes[0].set_title("Survival Time by Gender")

sns.boxplot(x='Stage', y='SurvivalTime', data=startingDF, ax=axes[1])
axes[1].set_title("Survival Time by Stage")

sns.boxplot(x='TreatmentType', y='SurvivalTime', data=startingDF, ax=axes[2])
axes[2].set_title("Survival Time by TreatmentType")

plt.suptitle("Box Plots of Categorical Features vs. SurvivalTime")
plt.show()

# %% [markdown]
# #### Point 4: Define the matrix X with the features as columns and examples as rows, and y as a vector with the Survival Time.

# %%
Remaining_X = startingDF_cleaned.drop(columns=['SurvivalTime', 'ID'])
Remaining_y = startingDF_cleaned['SurvivalTime']

# %%
Remaining_X

# %%
Remaining_y

# %%
X = DFnoNA[X_cols_universal]
y = DFnoNA[y_cols_universal]

# %%
X

# %%
y

# %% [markdown]
# #### Point 5: Consider a train, validation and test split, against a train, test split, with cross validation. What validation procedure is more data-efficient? Justify your answer with evidence from the dataset.
# 
# With a dataset of only 161 rows, it’s very important to use a validation procedure that maximizes data utilization. We will compare two approaches: Train, Validation and Test split without Cross-Validation and Train-Test split with Cross-Validation.   
# 
# - Train, Validation, and Test Split: In a typical train-validation-test split, for example 80-10-10, we would split approximately 129 rows for training, 16 rows for validation, and 16 rows for testing.
# The limitations can appear in the form of insufficient data in order to train a "decent" model. The small validation, in our case a validation set of 16 rows, may not provide reliable estimates for tuning hyperparameters (such as the degree in a polynomial regression, the alpha in a Ridge regression or the K in a KNN regression), as a small sample size can lead to high variance in validation performance.   
# In terms of data efficiency, this type of split is not very efficient for smaller datasets, as cutting aside separate validation and test sets reduces the number of samples available for training.
# 
# - Train-Test Split with Cross-Validation:
# In this approach, we split the data into 80% for training (about 129 rows) and 20% for testing (about 32 rows).
# An example of Cross-Validation is k-fold [[3]](#footnote3).   
# Cross-validation is more data-efficient because it allows the model to be trained and validated on different subsets of the training data without needing a separate validation set. It also provides a more reliable estimate of model performance by averaging results across folds, leading to better hyperparameter tuning and model selection. Not only that but this approach is highly data-efficient for smaller datasets like ours, as it maximizes the use of the available data for both training and validation while keeping a separate test set for unbiased evaluation.

# %%
startingDF_yCleaned = startingDF.dropna(subset=["SurvivalTime"])
X_universal = startingDF_yCleaned[X_cols_universal]
y_universal = startingDF_yCleaned[y_cols_universal]
survival_times = y_universal["SurvivalTime"]

df_survival_nan = startingDF[startingDF["SurvivalTime"].isna()]
X_universal_nan = df_survival_nan[X_cols_universal]
y_universal_nan = df_survival_nan[y_cols_universal]

# %%
plt.figure(figsize=(10, 6))
plt.hist(survival_times, bins=np.arange(1, max(survival_times) + 2), edgecolor='black', alpha=0.7)
plt.title('Histogram of Survival Times in train set', fontsize=14)
plt.xlabel('Survival Time (in months)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %% [markdown]
# #### Standard data splitting 

# %%
X_train, X_test, y_train, y_test = train_test_split(X_universal, y_universal, test_size=0.1, random_state=38)

survival_train = y_train["SurvivalTime"]
survival_test = y_test["SurvivalTime"]

print(f"The mean survival time in the training set is {survival_train.mean():.2f} months")
print(f"The mean survival time in the test set is {survival_test.mean():.2f} months")

palette = sns.color_palette("Set2")
plt.figure(figsize=(10, 6))
sns.kdeplot(survival_train, label="Train", fill=True, alpha=0.5, color=palette[0])
sns.kdeplot(survival_test, label="Test", fill=True, alpha=0.5, color=palette[1])
plt.legend()
plt.title("KDE Plot of Survival Times in Train and Test Sets")
plt.xlabel("Survival Time (in months)")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# %% [markdown]
# #### Global Datasets for Train/Test Validation using Stratified split

# %%
print("Shape of X_universal:", X_universal.shape)
print("Shape of y_universal:", y_universal.shape)

# %%
random_state = None

best_models = {}
#bin = np.quantile(survival_times, [0,0.25, 0.5, 0.75, 1])
#y_binned = pd.cut(survival_times.values.flatten(), bins=bin, labels=False, include_lowest=True)
#X_train, X_test, y_train, y_test = train_test_split(X_universal, y_universal, test_size=0.1, stratify=y_binned, random_state=random_state)



c0 = startingDF_yCleaned[startingDF_yCleaned["Censored"] == 0]
c1 = startingDF_yCleaned[startingDF_yCleaned["Censored"] == 1]
bins = [0, 2, 5, 6, 7, 11, 14]

y_binned_c0 = pd.cut(c0["SurvivalTime"].values.flatten(), bins=bins, labels=False, include_lowest=True)
y_binned_c1 = pd.cut(c1["SurvivalTime"].values.flatten(), bins=bins, labels=False, include_lowest=True)


X_train_c0, X_test_c0, y_train_c0, y_test_c0 = train_test_split(c0[X_cols_universal], c0[y_cols_universal], test_size=0.1, stratify=y_binned_c0, random_state=random_state)
X_train_c1, X_test_c1, y_train_c1, y_test_c1 = train_test_split(c1[X_cols_universal], c1[y_cols_universal], test_size=0.1, stratify=y_binned_c1, random_state=random_state)


X_train = pd.concat([X_train_c0, X_train_c1])
y_train = pd.concat([y_train_c0, y_train_c1])
X_test = pd.concat([X_test_c0, X_test_c1])
y_test = pd.concat([y_test_c0, y_test_c1])

X_train_NaN, X_test_NaN, y_train_NaN, y_test_NaN = train_test_split(X_universal_nan, y_universal_nan, test_size=0.1, random_state=random_state)

# %%
mean_censored_train = y_train['Censored'].mean()
mean_censored_test = y_test['Censored'].mean()

print(f"The mean of the SurvivalTime in the training set is {y_train['SurvivalTime'].mean():.2f}")
print(f"The mean of the SurvivalTime in the test set is {y_test['SurvivalTime'].mean():.2f}")

print(f"The mean of the censored values in the training set is {mean_censored_train:.2f}")
print(f"The mean of the censored values in the test set is {mean_censored_test:.2f}")

# %% [markdown]
# #### Analyzing the Stratified Split

# %%
survival_times = y_train["SurvivalTime"]
plt.figure(figsize=(10, 6))
plt.hist(survival_times, bins=np.arange(1, max(survival_times) + 2), edgecolor='black', alpha=0.7)
plt.title('Histogram of Survival Times in train set', fontsize=14)
plt.xlabel('Survival Time (in months)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
survival_times = y_test["SurvivalTime"]

plt.figure(figsize=(10, 6))
plt.hist(survival_times, bins=np.arange(1, max(survival_times) + 2), edgecolor='black', alpha=0.7)
plt.title('Histogram of Survival Times in test set', fontsize=14)
plt.xlabel('Survival Time (in months)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# %%
survival_train = y_train["SurvivalTime"]
survival_test = y_test["SurvivalTime"]

# Compute and print mean survival times
print(f"The mean survival time in the training set is {survival_train.mean():.2f} months")
print(f"The mean survival time in the test set is {survival_test.mean():.2f} months")

# Create KDE plot
palette = sns.color_palette("Set2")
plt.figure(figsize=(10, 6))
sns.kdeplot(survival_train, label="Train", fill=True, alpha=0.5, color=palette[0])
sns.kdeplot(survival_test, label="Test", fill=True, alpha=0.5, color=palette[1])
plt.legend()
plt.title("KDE Plot of Survival Times in Train and Test Sets")
plt.xlabel("Survival Time (in months)")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# %%
# Merge both sets
X_train = pd.concat([X_train, X_train_NaN])
y_train = pd.concat([y_train, y_train_NaN])
X_test = pd.concat([X_test, X_test_NaN])
y_test = pd.concat([y_test, y_test_NaN])

# Save this files in disk to be reused later
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# %% [markdown]
# #### Dropping the columns with missing values

# %%
X_train_reduced = X_train.drop(columns=missing_columns)
X_test_reduced = X_test.drop(columns=missing_columns)

# %%
# Number of NaN's in y_train and y_test
num_nan_y_train = y_train['SurvivalTime'].isna().sum()
num_nan_y_test = y_test['SurvivalTime'].isna().sum()

# Number of censored rows in y_train and y_test
num_censored_y_train = y_train['Censored'].sum()
num_censored_y_test = y_test['Censored'].sum()

print(f"Number of NaN's in y_train: {num_nan_y_train}")
print(f"Number of NaN's in y_test: {num_nan_y_test}")
print(f"Number of censored rows in y_train: {num_censored_y_train}")
print(f"Number of censored rows in y_test: {num_censored_y_test}")

# %%
train_mask = y_train["Censored"] == 0
X_train_filtered = X_train_reduced[train_mask]
y_train_filtered = y_train[train_mask]

test_mask = y_test["Censored"] == 0
X_test_filtered = X_test_reduced[test_mask]
y_test_filtered = y_test[test_mask]

# %%
# Number of NaN's in y_train and y_test
num_nan_y_train = y_train_filtered['SurvivalTime'].isna().sum()
num_nan_y_test = y_test_filtered['SurvivalTime'].isna().sum()

# Number of censored rows in y_train and y_test
num_censored_y_train = y_train_filtered['Censored'].sum()
num_censored_y_test = y_test_filtered['Censored'].sum()

print(f"Number of NaN's in y_train: {num_nan_y_train}")
print(f"Number of NaN's in y_test: {num_nan_y_test}")
print(f"Number of censored rows in y_train: {num_censored_y_train}")
print(f"Number of censored rows in y_test: {num_censored_y_test}")

# %%
X_train_filtered = X_train_filtered[y_train_filtered["SurvivalTime"].notna()]
y_train_filtered = y_train_filtered[y_train_filtered["SurvivalTime"].notna()]

X_test_filtered = X_test_filtered[y_test_filtered["SurvivalTime"].notna()]
y_test_filtered = y_test_filtered[y_test_filtered["SurvivalTime"].notna()]

# %%
# Number of NaN's in y_train and y_test
num_nan_y_train = y_train_filtered['SurvivalTime'].isna().sum()
num_nan_y_test = y_test_filtered['SurvivalTime'].isna().sum()

# Number of censored rows in y_train and y_test
num_censored_y_train = y_train_filtered['Censored'].sum()
num_censored_y_test = y_test_filtered['Censored'].sum()

print(f"Number of NaN's in y_train: {num_nan_y_train}")
print(f"Number of NaN's in y_test: {num_nan_y_test}")
print(f"Number of censored rows in y_train: {num_censored_y_train}")
print(f"Number of censored rows in y_test: {num_censored_y_test}")

# %%
survival_train = y_train_filtered["SurvivalTime"]
survival_test = y_test_filtered["SurvivalTime"]

# Compute and print mean survival times
print(f"The mean survival time in the training set is {survival_train.mean():.2f} months")
print(f"The mean survival time in the test set is {survival_test.mean():.2f} months")

# Create KDE plot
palette = sns.color_palette("Set2")
plt.figure(figsize=(10, 6))
sns.kdeplot(survival_train, label="Train", fill=True, alpha=0.5, color=palette[0])
sns.kdeplot(survival_test, label="Test", fill=True, alpha=0.5, color=palette[1])
plt.legend()
plt.title("KDE Plot of Survival Times in Train and Test Sets")
plt.xlabel("Survival Time (in months)")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# %% [markdown]
# ### Task 1.2 - Learn the baseline model

# %%
baseline_model = create_baseline_model(X_train_filtered, y_train_filtered["SurvivalTime"])
save_model(baseline_model, "baseline_reduced_features")
y_pred_baseline = baseline_model.predict(X_test_filtered)

cMSE_baseline = cMSE_error(y_test_filtered["SurvivalTime"], y_pred_baseline, y_test_filtered["Censored"])
print(f"cMSE (equivalent to MSE for non-censored data): {cMSE_baseline}")

# # Plot y vs. y-hat
plot_y_yhat(y_test_filtered["SurvivalTime"], y_pred_baseline, plot_title="y vs. y-hat (Baseline Linear Regression)")
plt.show()

# %%
baseline = ("Baseline (Linear Regression)", y_test_filtered["SurvivalTime"], y_pred_baseline, y_test_filtered["Censored"])
comparison_df = compare_models([baseline])
comparison_df.T

# %% [markdown]
# ##### Kaggle Prediction

# %%
kaggleDF_clean = kaggleDF.drop(columns=missing_columns)
KaggleSubmission = baseline_model.predict(kaggleDF_clean)
create_submission_csv(KaggleSubmission, "baseline_submission_09.csv")

# %% [markdown]
# ### Task 1.3 - Learn with the cMSE

# %% [markdown]
# Compute the expression of the derivative of the cMSE loss, where it is defined. Write your computations on the slides (can be a photo of handwritten math).    
# Now your training set will include the censored data that is not missing.

# %% [markdown]
# #### Derivative of Gradient for Loss Function
# 
# The loss function is defined as:
# 
# $$
# L(\alpha) = \frac{1}{N} \sum_{i=1}^{N} \Big( (1 - c)(y_i - \hat{y}_i)^2 + c \max(0, y_i - \hat{y}_i)^2 \Big)
# $$
# 
# where:
# $$
# \hat{y}_i = X_i^\top \alpha \text{ is the predicted values vector}
# $$
# 
# $$
# X \in \mathbb{R}^{d \times N} \text{ is the feature matrix.}
# $$
# 
# $$
# \alpha \in \mathbb{R}^d \text{ is the weight vector.}
# $$
# $$
# y \in \mathbb{R}^N \text{is the target vector.}
# $$
# 
# #### Compute the Derivative of the Loss function
# 
# $$
# \frac{\partial L}{\partial \alpha} = \frac{1}{N} \sum_{i=1}^N 
# \begin{cases} 
# 2(y_i - X_i^\top \alpha)X_i, & \text{if } y_i - X_i^\top \alpha > 0 \\
# 2(1 - c)(y_i - X_i^\top \alpha)X_i, & \text{if } y_i - X_i^\top \alpha \leq 0
# \end{cases}
# $$

# %%
def gradient_descent(X, y, c, iterations=1000, learning_rate=0.01, 
                     regularization=None, alpha=0.1, threshold=1e-6): 
    
    weights = np.random.rand(X.shape[1]) * 0.01  
    losses = []

    for i in range(iterations):
        # Compute predictions
        y_hat = X @ weights

        # Calculate cMSE loss
        loss = cMSE_error(y, y_hat, c)  
        losses.append(loss)

        # Check for convergence
        if i > 0 and abs(losses[-2] - losses[-1]) < threshold:  
            print(f"Convergence reached at iteration {i+1} with loss change below threshold.")
            break

        # Compute gradient
        grad_err = gradient_cMSE_error(y, y_hat, c, X)
        grad = -grad_err
        
        # Apply regularization
        if regularization == 'ridge':
            grad += 2 * alpha * weights  # Ridge (L2)
        elif regularization == 'lasso':
            grad += alpha * np.sign(weights)  # Lasso (L1)

        # Update weights
        weights -= learning_rate * grad

        # Print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i+1}: Loss = {loss}")

    return weights, losses

# %%
# Remove rows where the label is NaN in the training set
gd_X_train_reduced = X_train_reduced[y_train["SurvivalTime"].notna()]
gd_y_train_reduced = y_train[y_train["SurvivalTime"].notna()]

# Remove rows where the label is NaN in the test set
gd_X_test_reduced = X_test_reduced[y_test["SurvivalTime"].notna()]
gd_y_test_reduced = y_test[y_test["SurvivalTime"].notna()]

# %%
GDmodel_ridge = CustomLinearRegression(iterations=10000000, learning_rate=0.0000001, regularization='ridge', alpha=0.1)
GDmodel_ridge.fit(gd_X_train_reduced, gd_y_train_reduced)
save_model(GDmodel_ridge, "Gradient_Descent_model")

# %%
y_pred_gd_ridge = GDmodel_ridge.predict(gd_X_test_reduced)
gd_cMSE = cMSE_error(gd_y_test_reduced["SurvivalTime"], y_pred_gd_ridge, gd_y_test_reduced["Censored"])
print(f"cMSE for Gradient Descent: {gd_cMSE}")
print(f"cMSE for the Baseline Model: {cMSE_baseline}")

# %%
create_submission_csv(GDmodel_ridge.predict(kaggleDF_clean), "cMSE-baseline_submission-01.csv")

# %%
GDmodel_lasso = CustomLinearRegression(iterations=10000000, learning_rate=0.0000001, regularization='lasso', alpha=0.1)
GDmodel_lasso.fit(gd_X_train_reduced, gd_y_train_reduced)

y_pred_gd_lasso = GDmodel_lasso.predict(gd_X_test_reduced)
gd_cMSE_lasso = cMSE_error(gd_y_test_reduced["SurvivalTime"], y_pred_gd_lasso, gd_y_test_reduced["Censored"])
print(f"cMSE for Gradient Descent with Lasso: {gd_cMSE_lasso}")

# %%
GDmodel = CustomLinearRegression(iterations=10000000, learning_rate=0.0000001, regularization=None, alpha=0.1)
GDmodel.fit(gd_X_train_reduced, gd_y_train_reduced)

y_pred_gd = GDmodel.predict(gd_X_test_reduced)
gd_cMSE = cMSE_error(gd_y_test_reduced["SurvivalTime"], y_pred_gd, gd_y_test_reduced["Censored"])
print(f"cMSE for Gradient Descent: {gd_cMSE}")

# %%
df_GD = compare_models([("Baseline (Linear Regression)", y_test_filtered["SurvivalTime"], y_pred_baseline, y_test_filtered["Censored"]),
                        ("Gradient Descent", gd_y_test_reduced["SurvivalTime"], y_pred_gd, gd_y_test_reduced["Censored"]),
                        ("Gradient Descent (Ridge)", gd_y_test_reduced["SurvivalTime"], y_pred_gd_ridge, gd_y_test_reduced["Censored"]),
                        ("Gradient Descent (Lasso)", gd_y_test_reduced["SurvivalTime"], y_pred_gd_lasso, gd_y_test_reduced["Censored"])])

df_GD.T

# %%
plot_y_yhat(y_test_filtered["SurvivalTime"], y_pred_baseline, plot_title="y vs. y-hat (Baseline Linear Regression)")
plot_y_yhat(gd_y_test_reduced["SurvivalTime"], y_pred_gd, plot_title="y vs. y-hat (Gradient Descent)")
plot_y_yhat(gd_y_test_reduced["SurvivalTime"], y_pred_gd_lasso, plot_title="y vs. y-hat (Gradient Descent with Lasso)")
plot_y_yhat(gd_y_test_reduced["SurvivalTime"], y_pred_gd_ridge, plot_title="y vs. y-hat (Gradient Descent with Ridge)")

# %%



