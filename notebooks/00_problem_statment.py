# %% [markdown]
# # Assignment 2 - Semi-Supervised Learning for Predicting Survival Time in Multiple Myeloma Patients
# 
# [Kaggle Link](https://www.kaggle.com/competitions/machine-learning-nova-multiple-myeloma-survival)   
# [Notion Link](https://claudia-soares.notion.site/Assignment-2-12474b5ca0274b7bbb3c89c7dd5a5cf6#981cd24e241d42b4b77ee5e133fb4c6c)

# %% [markdown]
# ## Objective

# %% [markdown]
# 
# Apply semi-supervised techniques to develop predictive models for estimating the survival time of patients with [Multiple Myeloma](https://en.wikipedia.org/wiki/Multiple_myeloma).

# %% [markdown]
# ## Dataset Description

# %% [markdown]
# 
# - Synthetic simulation of clinical data for multiple myeloma patients, including variuous features with correlated missing values.    
# - Simulation of a commun real-world scenario by having an unlabeled "SurvivalTime" (complete information not always available) 
# - 9 features in the overall dataset:
#     - Age 
#     - Gender 
#     - Disease Stage
#     - Genetic Risk
#     - Treatment Type
#     - Comorbidity Index [[1]](#footnote1)
#     - Treatment Response
#     - Survival Time (Target)
#     - Censoring [[2]](#footnote2) Indicator
#         - Right censoring: a data point is above a certain value but it is unknown by how much.
# 
# Prediction Goal: The prediction goal is to corectly predict survival time from features, taking into account missing values in the features and missing labels (missing data == censored data). **Expect missing values in both sets**
# 

# %% [markdown]
# ### Features

# %% [markdown]
# | Column              | Description                                                                                                                                       | Type           |
# |---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
# | Age                 | Age of the patient. An integer number.                                                                                                            | Integer        |
# | Gender              | Biological sex of the patient.                                                                                                                    | Binary         |
# | Stage               | Extent of the cancer, ranging from 1 (less severe) to 4 (very serious).                                                                            | Integer (1-4)  |
# | GeneticRisk         | Combined information on a patient's genetic cancer risk, ranging between 0 and 1.                                                                  | Real (0-1)     |
# | TreatmentType       | Type of treatment administered (more aggressive or milder).                                                                                       | Binary         |
# | ComorbidityIndex    | Quantifies the number and severity of additional comorbid conditions. Integer where 0 indicates no comorbid conditions.                            | Integer        |
# | TreatmentResponse   | Effectiveness of the treatment administered (0 for poor/inadequate response).                                                                     | Binary         |

# %% [markdown]
# ### Label and Censored Indicator

# %% [markdown]
# | Column        | Description                                                                                                                                   | Type       |
# |---------------|-----------------------------------------------------------------------------------------------------------------------------------------------|------------|
# | SurvivalTime  | Duration from the start of the study or treatment to the event of interest (e.g., death, disease progression), or until the last follow-up.     | Continuous |
# | Censored      | Indicates whether the survival time is censored (1 for censored, meaning the event did not occur by the end of the study or follow-up).         | Binary     |
# 


