from matplotlib import pyplot
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.lines as mlines
import torch
import numpy as np


# Take in x tensor, y array and model
# Output overall accuracy
def accuracy(x, y, model):
    test = model(x)

    accuracy = 0
    testSetSize = len(test)
    # Shape: [450, 1, 4097]

    # Iterate accuracy whenever a correct prediction is made
    for index in range(testSetSize):
        if (test[index][0].data.numpy()[0] > test[index][0].data.numpy()[1]) and y[index][0] == 1:
            accuracy += 1
        elif (test[index][0].data.numpy()[0] < test[index][0].data.numpy()[1]) and y[index][1] == 1:
            accuracy += 1

    print("Accuracy: " + str(accuracy) + "/" + str(testSetSize) + " = " + str(accuracy / testSetSize) + "%")


# Take in x tensor, y outputs and model
# Outputs an ROC curve
def generateROC(x, y, model):
    predictions = model(x)

    # Get all values for ictal and healthy predictions
    ictal_outcomes = predictions.squeeze(1).data.numpy()[:, 1]
    healthy_outcomes = predictions.squeeze(1).data.numpy()[:, 0]

    # Get all ictal and healthy expected values
    y_ictal = y[:, 1]
    y_healthy = y[:, 0]

    # Calculate ROC AUC
    ictal_auroc = roc_auc_score(y_ictal, ictal_outcomes)
    healthy_auroc = roc_auc_score(y_healthy, healthy_outcomes)

    print('ICTAL ROC AUC=%.3f' % ictal_auroc)
    print('HEALTHY ROC AUC=%.3f' % healthy_auroc)

    # Calculate false positive, true positive rates
    ictal_fpr, ictal_tpr, _ = roc_curve(y_ictal, ictal_outcomes)
    healthy_fpr, healthy_tpr, _ = roc_curve(y_healthy, healthy_outcomes)

    # Plot healthy and ictal ROCs and plot baseline
    pyplot.plot(ictal_fpr, ictal_tpr, marker='.', label='Ictal (AUC =%.3f' % ictal_auroc + ')')
    pyplot.plot(healthy_fpr, healthy_tpr, marker='.', label='Healthy (AUC=%.3f' % healthy_auroc + ')')
    pyplot.plot([0.0, 1.0], linestyle='--', label='Baseline')

    # Label axes
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.title('EEG Classifier ROC Curve (' + str(len(y)) + " Samples)")

    # Show plot
    pyplot.legend()
    pyplot.show()

    # Show test set size
    print("y size: " + str(len(y)))


