from imblearn.over_sampling import RandomOverSampler

def random_upsampling(X, y):
    X = X.reshape([X.shape[0],-1])
    print(X.shape)
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res