from prediction.data_preprocessing import final_loading
from prediction.model_training import model_training
from prediction.model_evaluation import evaluate,final_submission

data='data/WA_Fn-UseC_-Telco-Customer-Churn (1).csv'

def main(data):
    x_train,x_test,y_train,y_test=final_loading(data)
    model=model_training(x_train,y_train)

    evaluate(model,x_train,y_train)

    y_pred=model.predict(x_test)

    final_submission(x_test,y_test,y_pred,output_path='submission.csv')

