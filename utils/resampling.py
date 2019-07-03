from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def random_upsampling(X, y, random_state):
    shape = X.shape
    X = X.reshape([shape[0], -1])
    ros = RandomOverSampler(random_state=random_state)
    X_res, y_res = ros.fit_resample(X, y)
    X_res = X_res.reshape([X_res.shape[0]] + list(shape[1:]))
    y_res = y_res.reshape([-1, 1])
    return X_res, y_res


def random_downsampling(X, y, random_state):
    shape = X.shape
    X = X.reshape([shape[0], -1])
    rus = RandomUnderSampler(random_state=random_state)
    X_res, y_res = rus.fit_resample(X, y)
    X_res = X_res.reshape([X_res.shape[0]] + list(shape[1:]))
    y_res = y_res.reshape([-1, 1])
    return X_res, y_res


def multi_inputs_random_upsampling(X1, X2, y, random_state):
    shape1 = X1.shape
    X1 = X1.reshape([shape1[0], -1])
    shape2 = X2.shape
    X2 = X2.reshape([shape2[0], -1])
    ros = RandomOverSampler(random_state=random_state)
    X1_res, y_res = ros.fit_resample(X1, y)
    X1_res = X1_res.reshape([X1_res.shape[0]] + list(shape1[1:]))
    X2_res, y_res = ros.fit_resample(X2, y)
    X2_res = X2_res.reshape([X2_res.shape[0]] + list(shape2[1:]))
    return X1_res, X2_res, y_res
