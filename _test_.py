
import torch
import torchvision

import layers
import status
import utils

def test_layers():
    x = torch.randn(3, 64, 224, 224)
    m = layers.ConvBatchnormRelu2d(64, 32, 3, padding=1)
    x = m(x)
    m = layers.Flatten()
    x = m(x)
    m = layers.SingSqrt()
    x = m(x)
    m = layers.L2Norm()
    x = m(x)


def test_status(figs=False):
    import time

    c_status = status.Classification()
    r_status = status.Regression()
    g_status = status.GAN()

    start = time.time()
    for index in range(10):
        for batch_index in range(10):
            g_status.append(index, batch_index)

        c_status.append_train(index, batch_index)
        c_status.append_validation(batch_index, index)
        r_status.append_train(index)
        r_status.append_validation(batch_index)

        print(c_status.should_save())
        print(r_status.should_save())

        c_status.verbose(10, sec=time.time()-start)
        r_status.verbose(10, sec=time.time()-start)
        g_status.verbose(10, sec=time.time()-start)

    if figs:
        c_status.plot('./c.png')
        r_status.plot('./r.png')
        g_status.plot('./g.png')

def test_utils(figs=False):
    label_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    predicted = [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
    target    = [0, 0, 2, 3, 4, 5, 6, 0, 1, 1, 3, 4, 5, 6, 0, 1, 2, 2, 4, 5, 6]

    if figs:
        utils.plot_confusion_matrix(
            predicted, target, label_names
        )
    predicted, target = utils.label2name(
        predicted, target, label_names
    )
    print(predicted)
    print(target)


if __name__ == "__main__":
    figs = False

    test_layers()
    test_status(figs)
    test_utils(figs)