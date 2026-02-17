import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(y_true, y_pred, class_names):
    """
    Print classification report and confusion matrix
    """
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12,8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def top5_accuracy(y_true, y_pred_probs):
    """
    Calculate Top-5 accuracy
    """
    top5 = np.argsort(y_pred_probs, axis=1)[:, -5:]
    correct = 0

    for i in range(len(y_true)):
        if y_true[i] in top5[i]:
            correct += 1

    return correct / len(y_true)
