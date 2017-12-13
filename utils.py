import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

def show_losses(histories, acc='acc'):
    plt.figure(figsize=(6,6))
    #plt.ylim(bottom=0)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Error by Epoch')
    colors=[]
    do_acc=False
    for label,loss in histories:
        color = tuple(np.random.random(3))
        colors.append(color)
        l = label
        vl= label+" validation"
        if acc in loss.history:
            l+=' (%s %2.4f)'% (acc, loss.history[acc][-1])
            do_acc = True
        if 'val_'+acc in loss.history:
            vl+=' (val_%s %2.4f)'% (acc,loss.history['val_'+acc][-1])
            do_acc = True
        plt.plot(loss.history['loss'], lw=4, label=l, color=color)
        if 'val_loss' in loss.history:
            plt.plot(loss.history['val_loss'], lw=4, ls='dashed', label=vl, color=color)
    plt.legend(loc='best')
    #plt.yscale('log')
    plt.show()
    
    if not do_acc: return
    plt.figure(figsize=(6,6))
    plt.xlabel('Epoch')
    plt.ylabel(acc)
    for i,(label,loss) in enumerate(histories):
        color = colors[i]
        if acc in loss.history:
            plt.plot(loss.history[acc], lw=4, label=label, color=color)
        if 'val_'+acc in loss.history:
            plt.plot(loss.history['val_'+acc], lw=4, ls='dashed', label=label + ' validation', color=color)
    plt.legend(loc='best')
    plt.show()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          size=(6.,6.),
                          show_label=True,
                          show_num=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize=size)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' #if normalize else '.0f'
    thresh = cm.max() / 2.
    if show_num:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    if show_label:
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    
    # set color bar same length with fig size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def plot_correlation_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.RdBu,
                          size=(6.,6.),
                          show_num=True,
                          elev_min=-1.,
                          elev_max=1.):

    #print(cm)

    plt.figure(figsize=size)
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap,clim=(elev_min, elev_max))
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' 
    thresh = cm.max() / 2.
    if show_num:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     verticalalignment="center",
                     fontsize=14,
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # set color bar same length with fig size
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


