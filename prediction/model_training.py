from sklearn.linear_model import LogisticRegression

def model_training(x_train,y_train):
    model=LogisticRegression(max_iter=10000,class_weight='balanced')
    model.fit(x_train,y_train)

    return model