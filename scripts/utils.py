import sys
import pandas as pd
from joblib import dump, load

X_cols_universal = ['Age', "Gender", "Stage", "GeneticRisk", "TreatmentType", "ComorbidityIndex", "TreatmentResponse"]
y_cols_universal = ["SurvivalTime", "Censored"]
missing_columns = ['GeneticRisk', 'ComorbidityIndex', 'TreatmentResponse']

def create_submission_csv(predictions, filename, path="results/submissions/"):
    column_mapping = {
        0: '0'
    }
    
    predictions_df = pd.DataFrame(predictions)
    predictions_df.rename(columns=column_mapping, inplace=True)
    predictions_df.insert(0, 'id', range(len(predictions_df)))
    predictions_df.to_csv(path+filename, index=False)
    
    print(f"CSV file '{filename}' has been created.")
    
def load_starting_df():
    #print("Updated Python Path:", sys.path)
    return pd.read_csv("data/train_data.csv", sep=",").rename(columns={"Unnamed: 0":"ID"})

def load_kaggle_df():
    return pd.read_csv("data/test_data.csv", sep=",").drop(columns=["id"])

def load_train_test():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv")
    y_test = pd.read_csv("data/y_test.csv")
    
    return X_train, X_test, y_train, y_test
    

def save_model(model, name):
    dump(model, "models/"+name+".joblib")
    
def load_model(name):
    return load("models/"+name+".joblib")