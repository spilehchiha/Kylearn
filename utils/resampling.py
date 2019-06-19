from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def random_upsampling(X, y, random_state):
    shape = X.shape
    X = X.reshape([shape[0],-1])
    ros = RandomOverSampler(random_state = random_state)
    X_res, y_res = ros.fit_resample(X, y)
    X_res = X_res.reshape([X_res.shape[0]]+list(shape[1:]))
    y_res = y_res.reshape([-1,1])
    return X_res, y_res

def random_downsampling(X, y, random_state):
    shape = X.shape
    X = X.reshape([shape[0], -1])
    rus = RandomUnderSampler(random_state = random_state)
    X_res, y_res = rus.fit_resample(X, y)
    X_res = X_res.reshape([X_res.shape[0]] + list(shape[1:]))
    y_res = y_res.reshape([-1, 1])
    return X_res, y_res
