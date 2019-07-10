import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import classification_report
def metrics_binary(y_pred, threshold, y_test):
    y_pred_copy = y_pred.copy()
    y_pred_copy[y_pred_copy >= threshold] = 1
    y_pred_copy[y_pred_copy < threshold] = 0
    cm = confusion_matrix(y_test, y_pred_copy)
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (np.sum(cm))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('accuracy: ' + str(accuracy) + '\n')
    print('precision: ' + str(precision) + '\n')
    print('recall: ' + str(recall) + '\n')
    print('false positive rate: ' + str(fpr) + '\n')
    return cm, fpr, accuracy, precision, recall

def metrics_multi(y_pred, y_test, labels):
    cm = confusion_matrix(np.array(y_test).flatten(), np.array(y_pred).flatten())
    print(classification_report(np.array(y_test).flatten(), np.array(y_pred).flatten()))
    acc = accuracy_score(y_test, y_pred)
    return cm, acc


def auc_roc(y_pred, y_test):
    auc = roc_auc_score(y_true=y_test, y_score=y_pred)
    fprs, tprs, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    return auc, fprs, tprs, thresholds