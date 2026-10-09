import pandas as pd

def evaluate(model,x_train,y_train):
    score =model.score(x_train,y_train)
    return print(f"model score is : {score}")

def final_submission(x_test,y_test,y_pred,output_path='submission.csv'):
    submission=pd.DataFrame({
        'Churn':y_pred
    })
    submission.to_csv(index=True)
    print(f"submission completed and saved to {output_path}")
    return submission