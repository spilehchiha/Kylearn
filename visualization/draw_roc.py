from matplotlib import pyplot

def plot_roc_curve(fprs, tprs):
    pyplot.plot([0,1],[0,1], linestyle='--')
    pyplot.plot(fprs, tprs, marker='.')
    pyplot.show()