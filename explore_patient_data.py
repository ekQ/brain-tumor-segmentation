import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
import sys

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

def plot_scatter_matrix(x,y,fname='plots/scatter_matrix.png'):
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
        plot_scatter_matrix(x,y,fname='plots/scatter_matrix_%s.png' % modalities[i])

def plot_predictions(coord, dim, pred, gt):
    assert coord.shape[0] == len(pred), "Number of coordinates must match to the number of labels (%d != %d)" % (coord.shape[0], len(pred))
    print "Plotting predictions..."
    D = np.ones((dim[0], dim[1], dim[2])) * -1
    for i in range(coord.shape[0]):
        D[coord[i,0], coord[i,1], coord[i,2]] = pred[i]
    Dgt = np.ones((dim[0], dim[1], dim[2])) * -1
    for i in range(coord.shape[0]):
        Dgt[coord[i,0], coord[i,1], coord[i,2]] = gt[i]

    n_layers = 5
    zs = np.linspace(0, dim[2], n_layers+4)
    zs = map(int, zs[2:-2])
    plt.figure(2)
    from matplotlib import colors
    cmap = colors.ListedColormap(['white', 'orange', 'red', 'blue', 'green', 'black'])
    bounds = range(-1,6)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    for i in range(n_layers):
        plt.subplot(2,n_layers, i+1)
        plt.imshow(D[:,:,zs[i]], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
        plt.title('Prediction (z=%d)' % zs[i])
        plt.subplot(2,n_layers, i+1+n_layers)
        plt.imshow(Dgt[:,:,zs[i]], interpolation='nearest', origin='lower', cmap=cmap, norm=norm)
        plt.title('Ground truth (z=%d)' % zs[i])
    plt.show()

def save_predictions(coord, dim, pred, gt, fname='predictions/pred.csv'):
    print 'Writing predictions to %s...' % fname
    with open(fname, 'w') as fout:
        fout.write('x,y,z,pred,ground_truth\n')
        for i in range(coord.shape[0]):
            fout.write('%d,%d,%d,%d,%d\n' % (coord[i,0], coord[i,1], coord[i,2], pred[i], gt[i]))
    with open(fname+'.dim', 'w') as fout2:
        fout2.write('%d,%d,%d\n' % (dim[0], dim[1], dim[2]))
    print 'Written.'

def load_patient(number, do_preprocess=True):
    data = scipy.io.loadmat('data/Patient_Features_%d.mat' % number)
    data = data['featuresMatrix']

    tumor_grade = data[0,0]
    print "Patient %d, tumor grade: %d" % (number, tumor_grade)

    row0 = 5
    y = data[row0:, 1]
    x = data[row0:, 5:]
    #x = data[row0:, 5:11]
    #x = data[row0:, [5,11,17,23]]
    coord = data[row0:, 2:5]
    dim = data[3, :3]

    if do_preprocess:
        x = preprocess(x)

    return x, y, coord, dim

def preprocess(x):
    # Median to zero
    x -= np.median(x,0)
    # Variance to 1
    x /= np.std(x,0)
    return x

def dice(y, ypred):
    A = set(np.nonzero(y)[0])
    B = set(np.nonzero(ypred)[0])
    score = 2*len(A & B) / float(len(A) + len(B))
    return score

def dice_scores(y, ypred, patient_idxs=None):
    if patient_idxs is None:
        patient_idxs = [0, len(y)]
    ds = np.zeros((len(patient_idxs)-1,3))
    for i in range(len(patient_idxs)-1):
        yy = y[patient_idxs[i]:patient_idxs[i+1]]
        yypred = ypred[patient_idxs[i]:patient_idxs[i+1]]
        ds[i,0] = dice(yy>0, yypred>0)
        ds[i,1] = dice(np.logical_and(yy>0, yy!=2),
                       np.logical_and(yypred>0, yypred!=2))
        ds[i,2] = dice(yy==4, yypred==4)
    return ds

np.random.seed(1337)

patients = np.random.permutation(50) + 1
n_tr_p = 2
n_te_p = 1
train_patients = patients[:n_tr_p]
test_patients = patients[n_tr_p:n_tr_p+n_te_p]

n_columns = 24
xtr = np.zeros((0,n_columns))
ytr = np.zeros(0)
print "Train users..."
for tr_pat in train_patients:
    x, y, _, _ = load_patient(tr_pat)
    idxs = np.random.permutation(len(y))
    idxs = idxs[:1000]
    ytr = np.concatenate((ytr, y[idxs]))
    xtr = np.vstack((xtr, x[idxs,:]))

#plot_all_scatter_matrices(xtr, ytr)

xte = np.zeros((0,n_columns))
yte = np.zeros(0)
patient_idxs = [0]
print "Test users..."
for te_pat in test_patients:
    x, y, coord, dim = load_patient(te_pat)
    idxs = np.random.permutation(len(y))
    idxs = idxs#[:100000]
    yte = np.concatenate((yte, y[idxs]))
    xte = np.vstack((xte, x[idxs,:]))
    patient_idxs.append(len(yte))
    coord = coord[idxs,:]
n_non_zeros = sum(yte > 0)
print "%.1f%% tumor voxels (total %d)" % (100.0 * n_non_zeros / float(len(yte)), len(yte))

# Classify
n_trees = 10
t0 = time.time()
model = RandomForestClassifier(n_trees, oob_score=True, verbose=1, n_jobs=4)
#model = ExtraTreesClassifier(n_trees, verbose=1, n_jobs=4)
#model = svm.SVC(C=1000)
model.fit(xtr, ytr)
print "Training took %.2f seconds" % (time.time()-t0)
#print "OOB score: %.2f%%" % (model.oob_score_*100)
print "Feature importances:"
for i in range(4):
    print model.feature_importances_[i*6:(i+1)*6]

pred = model.predict(xte)
print "\nConfusion matrix:"
cm = confusion_matrix(yte, pred)
print cm

acc = sum(pred==yte) / float(len(pred))
bl_acc = sum(yte==0) / float(len(pred))
print "Accuracy:\t%.2f%%" % (acc*100)
print "Majority vote:\t%.2f%%" % (bl_acc*100)

ds = dice_scores(yte, pred, patient_idxs=patient_idxs)
ds_mean = np.mean(ds,0)
ds_std = np.std(ds,0)
ds_min = np.min(ds,0)
ds_max = np.max(ds,0)
print "\nDice scores:"
print "              \tMean\tStd\tMin\tMax"
print "Whole tumor:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[0], ds_std[0], ds_min[0], ds_max[0])
print "Tumor core:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[1], ds_std[1], ds_min[1], ds_max[1])
print "Active tumor:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[2], ds_std[2], ds_min[2], ds_max[2])

#plot_predictions(coord, dim, pred, yte)
save_predictions(coord, dim, pred, yte, 'predictions/pred_patient%d.csv' % test_patients[0])

#plt.figure()
#plot_confusion_matrix(cm)
#plt.show()
#"""
