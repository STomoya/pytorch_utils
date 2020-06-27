
import torch
import torchvision.transforms as T
import numpy as np


'''
ImageNet normalize transform
'''
ImageNetNormalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def plot_confusion_matrix(
    predicted, target, label_names, filename='./cm.png',
    is_names=False, cmap='Blues', normalize='true',
):
    '''
    plot confusion matrix
    convert labels to their names if needed

    arg
        predicted: list of int or str
            predicted labels or names
        target: list of int or str
            target labels or names
        label_names: list of str
            list of names of labels
        filename: str (default: ./cm.png)
            filename for output picture of confusion matrix
        is_name: bool (default: False)
            if True, will skip index to name conversion
        cmap: str (default: 'Blues')
            color map of matplotlib
        normalize: str (default: 'true')
            'true' -> row, 'pred' -> column,
            'all'  -> all, None   -> no normalization
    '''
    
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    if not is_names:
        predicted, target = label2name(
            predicted, target, label_names
        )

    predicted = np.array(predicted).reshape(-1, 1)
    target    = np.array(target).reshape(-1, 1)

    cm = confusion_matrix(
        predicted,
        target,
        labels=label_names,
        normalize=normalize
    )

    ConfusionMatrixDisplay(cm, display_labels=label_names).plot(cmap=cmap)
    plt.savefig(filename)


def label2name(
    predicted,
    target,
    label_names
):
    '''
    convert labels to target

    arg
        predicted: list of int
            predicted labels
        target: list of int
            target labels
        label_names: list of str
            list of names of labels
    '''

    predicted_names = []
    target_names    = []

    for p, t in zip(predicted, target):
        predicted_names.append(label_names[p])
        target_names.append(label_names[t])

    return predicted_names, target_names
