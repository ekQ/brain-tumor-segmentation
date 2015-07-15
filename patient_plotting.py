import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import cPickle as pickle

def plot_cm(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    #tick_marks = np.arange(len(iris.target_names))
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix(cm):
    plt.subplot(121)
    #plot_cm(np.log(cm), title='Confusion matrix (log scale)')
    plot_cm(cm, title='Confusion matrix')

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.subplot(122)
    plot_cm(cm_normalized, title='Normalized confusion matrix')

def plot_scatter_matrix(x,y,fname='scatter_matrix.png'):
    import pandas as pd
    from pandas.tools.plotting import scatter_matrix
    df = pd.DataFrame(np.hstack((x,y.reshape(len(y),1))), columns=['intensity', 'gaussian', 'gradient mag', 'grad dir', 'laplacian', 'imglog', 'label'])
    df['label'] = df['label'].astype(int)
    colors = ['red','green','blue','cyan','black']
    import matplotlib.pyplot as plt
    scatter_matrix(df,figsize=[9,7],marker='x',c=df.label.apply(lambda xx:colors[xx]))
    plt.savefig(fname)
    print "Saved scatter matrix to %s" % fname

def plot_all_scatter_matrices(xx, yy):
    modalities = ['FLAIR', 'T1C', 'T1', 'T2']
    for i in range(4):
        idxs = range(i*6,(i+1)*6)
        x = xx[:,idxs]
        y = yy
        plot_scatter_matrix(x,y,fname=os.path.join('plots', 'scatter_matrix_%s.png' % modalities[i]))

def plot_predictions(coord, dim, pred, gt=None, pp_pred=None, fname=None, fpickle=None):
    assert coord.shape[0] == len(pred), "Number of coordinates must match to the number of labels (%d != %d)" % (coord.shape[0], len(pred))
    print "Plotting predictions..."
    D = np.ones((dim[0], dim[1], dim[2])) * -1
    for i in range(coord.shape[0]):
        D[coord[i,0], coord[i,1], coord[i,2]] = pred[i]
    if gt is not None:
        Dgt = np.ones((dim[0], dim[1], dim[2])) * -1
        for i in range(coord.shape[0]):
            Dgt[coord[i,0], coord[i,1], coord[i,2]] = gt[i]
    if pp_pred is not None:
        Dpp = np.ones((dim[0], dim[1], dim[2])) * -1
        for i in range(coord.shape[0]):
            Dpp[coord[i,0], coord[i,1], coord[i,2]] = pp_pred[i]
    n_layers = 7
    n_rows = 1
    if gt is not None:
        n_rows += 1
    if pp_pred is not None:
        n_rows += 1
    zs = np.linspace(0, dim[2], n_layers+4)
    zs = map(int, zs[2:-2])
    plt.figure(2, figsize=(n_layers*4,10))
    cmap = colors.ListedColormap(['white', 'orange', 'red', 'blue', 'green', 'black'])
    bounds = range(-1,6)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    for i in range(n_layers):
        row_idx = 0
        if gt is not None:
            plt.subplot(n_rows,n_layers, i+1)
            plt.imshow(Dgt[:,:,zs[i]], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            plt.title('Ground truth (z=%d)' % zs[i])
            row_idx += 1
        plt.subplot(n_rows,n_layers, i+1+(row_idx*n_layers))
        plt.imshow(D[:,:,zs[i]], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
        plt.title('Prediction (z=%d)' % zs[i])
        row_idx += 1
        if pp_pred is not None:
            plt.subplot(n_rows,n_layers, i+1+(row_idx*n_layers))
            plt.imshow(Dpp[:,:,zs[i]], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
            plt.title('Post-processed (z=%d)' % zs[i])
            
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)
    if fpickle is not None:
        with open(fpickle, 'wb') as fp:
            pickle.dump(D, fp)
    print "Done.\n"

def save_pred_probs_csv(coord, dim, pred_probs, fname):
    X = np.hstack((coord, pred_probs))
    header = np.zeros((1,X.shape[1]))
    header[0,:3] = dim
    X = np.vstack((header, X))
    np.savetxt(fname, X)
    print "Saved %s" % fname
