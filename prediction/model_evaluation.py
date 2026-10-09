import pandas as pd

def evaluate(model,x_train,y_train):
    score =model.score(x_train,y_train)
    return print(f"model score is : {score}")

def final_submission(y_pred)