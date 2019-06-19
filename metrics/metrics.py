import numpy as np
def fpr_metrics(cm):
    tp = cm[1, 1]
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (np.sum(cm))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print('false positive rate: ' + str(fpr) + '\n')
    print('accuracy: ' + str(acc) + '\n')
    print('precision: ' + str(precision) + '\n')
    print('recall: ' + str(recall) + '\n')

def cm_metrix(y_test, y_pred):
    return confusion_matrix(y_true=y_test, y_pred=y_pred)