import warnings
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Suppress sklearn metric warnings
warnings.filterwarnings('ignore')


def precision_recall_f1(true_labels, predicted_labels, labels):
    """Generates classification metrics precision, recall and F1 for the given predictions.

    Args:
        true_labels (np.array): The ground truth labels for the data
        predicted_labels (np.array): The predicted labels for the data
        labels (list): List of label names

    Returns:
        metrics_dict (dict): Dictionary of micro, macro and weighted precision, recall and F1 scores
        metrics_str (str): Formatted string with metric data for all classes and totals
    """

    assert len(true_labels) == len(predicted_labels), "True labels " + str(len(true_labels)) + \
                                                      " doesn't match predicted labels " + str(len(predicted_labels))
    # Initialise test metrics dictionary
    metrics_dict = {}

    # Calculate metrics globally by counting the total true positives, false negatives and false positives
    precision_mic, recall_mic, f1_mic, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                           average='micro')
    metrics_dict['precision_micro'] = precision_mic
    metrics_dict['recall_micro'] = recall_mic
    metrics_dict['f1_micro'] = f1_mic

    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account
    precision_mac, recall_mac, f1_mac, support = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                                 average='macro')
    metrics_dict['precision_macro'] = precision_mac
    metrics_dict['recall_macro'] = recall_mac
    metrics_dict['f1_macro'] = precision_mac

    # Calculate metrics for each label, and find their average weighted by support (number of true instances for each label)
    precision_weight, recall_weight, f1_weight, _ = precision_recall_fscore_support(true_labels, predicted_labels,
                                                                                    average='weighted')
    metrics_dict['precision_weighted'] = precision_weight
    metrics_dict['recall_weighted'] = recall_weight
    metrics_dict['f1_weighted'] = f1_weight

    # Need to remove labels that are not present in the test set at all
    for i in range(len(labels)):
        if i not in true_labels:
            del labels[i]
    # Generate classification report for each label/totals
    metric_str = classification_report(true_labels, predicted_labels, target_names=labels)

    return metrics_dict, metric_str


def plot_confusion_matrix(true_labels, predicted_labels, labels, title=None, matrix_dim=15,
                          normalise=True, fig_size=(10, 10), font_size=15):
    """Generates a confusion matrix for the given predictions.

    Uses sklearn to generate the confusion matrix.
    Uses matplotlib and seaborn to generate the figure.

    Args:
        true_labels (np.array): The ground truth labels for the data
        predicted_labels (np.array): The predicted labels for the data
        labels (list): List of label names
        title (str): The title for the figure
        matrix_dim (int): The number of classes to show on the matrix, if None or -1 creates full matrix
        normalise (bool): Whether to normalise the matrix or use original values
        fig_size (tuple): Tuple for the horizontal and vertical size the figure
        font_size (int): Font size for figure labels.

    Returns:
        fig (matplotlib figure): The confusion matrix figure
    """

    # Generate the confusion matrix array
    cm = confusion_matrix(true_labels, predicted_labels)

    # Normalise the matrix
    if normalise:
        # Ignore divide by zero for no predictions for a certain label
        with np.errstate(divide='ignore', invalid='ignore'):
            cm = np.true_divide(cm.astype('float'), cm.sum(axis=1)[:, np.newaxis])
            cm[~ np.isfinite(cm)] = 0  # -inf inf NaN
        fmt = '.2f'
    else:
        fmt = 'd'

    # Truncate matrix and labels to desired matrix dimensions
    if matrix_dim:
        cm = cm[:matrix_dim, :matrix_dim]
        labels = labels[:cm.shape[0]]

    # Create pandas dataframe
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    # Create figure and heatmap
    fig = plt.figure(figsize=fig_size)
    heatmap = sns.heatmap(df_cm, cmap="YlGnBu", annot=True, fmt=fmt, linewidths=0.5, cbar=False)

    # Set labels
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=font_size)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=font_size)
    plt.ylabel('True label', fontsize=font_size)
    plt.xlabel('Predicted label', fontsize=font_size)
    if title:
        plt.title(title, fontsize=font_size)

    return fig
