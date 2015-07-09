import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import time
import matplotlib.pyplot as plt
import sys
import os.path

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

def plot_predictions(coord, dim, pred, gt, fname=None):
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
    plt.figure(2, figsize=(20,10))
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
    if fname is None:
        plt.show()
    else:
        plt.savefig(fname)

def save_predictions(coord, dim, pred, gt, fname='pred.csv'):
    print 'Writing predictions to %s...' % fname
    with open(fname, 'w') as fout:
        fout.write('x,y,z,pred,ground_truth\n')
        for i in range(coord.shape[0]):
            fout.write('%d,%d,%d,%d,%d\n' % (coord[i,0], coord[i,1], coord[i,2], pred[i], gt[i]))
    with open(fname+'.dim', 'w') as fout2:
        fout2.write('%d,%d,%d\n' % (dim[0], dim[1], dim[2]))
    print 'Written.'

def load_patient(number, do_preprocess=True, n_voxels=None):
    data = scipy.io.loadmat(os.path.join('data', 'Patient_Features_%d.mat' % number))
    data = data['featuresMatrix']

    tumor_grade = data[0,0]
    print "Patient %d, tumor grade: %d" % (number, tumor_grade)

    row0 = 5
    y = data[row0:, 1]
    x = data[row0:, 5:]
    #x = data[row0:, 5:11]
    #x = data[row0:, [5,11,17,23]]
    if do_preprocess:
        x = preprocess(x)
    coord = data[row0:, 2:5]
    if n_voxels is not None and isinstance(n_voxels, int):
        idxs = np.random.permutation(len(y))
        idxs = idxs[:min(n_voxels,len(y))]
        y = y[idxs]
        x = x[idxs,:]
        coord = coord[idxs,:]

    dim = data[3, :3]

    return x, y, coord, dim

def preprocess(x):
    # Median to zero
    x -= np.median(x,0)
    # Variance to 1
    x /= np.std(x,0)
    return x

def extract_label_features(coords, dims, pred_probs, patient_idxs):
    """
    Return histogram of predicted labels in the neighborhood for each voxel.
    """
    t0 = time.time()
    print "Extracting label features..."
    n_modalities = 5
    xlabel = np.zeros((coords.shape[0], n_modalities))
    # Neighborhood radius
    r = 1
    # Neighborhood patch
    patch = np.ones((2*r+1, 2*r+1, 2*r+1))
    patch = patch[np.newaxis,:,:,:]
    # Go through patients
    for pi in range(len(patient_idxs)-1):
        pidxs = range(patient_idxs[pi], patient_idxs[pi+1])
        pcoords = coords[pidxs,:]
        ppred_probs = pred_probs[pidxs,:]
        dim = dims[pi]
        # Histogram of predicted labels for neighbors
        label_hist = np.zeros((n_modalities, dim[0], dim[1], dim[2]))
        for i in range(pcoords.shape[0]):
            coord = pcoords[i,:]
            probs = ppred_probs[i,:]
            x0 = max(coord[0]-r, 0)
            x1 = min(coord[0]+r, dim[0])
            dx0 = x0 - coord[0]
            dx1 = x1 - coord[0]
            y0 = max(coord[1]-r, 0)
            y1 = min(coord[1]+r, dim[1])
            dy0 = y0 - coord[1]
            dy1 = y1 - coord[1]
            z0 = max(coord[2]-r, 0)
            z1 = min(coord[2]+r, dim[2])
            dz0 = z0 - coord[2]
            dz1 = z1 - coord[2]

            # Go through modalities
            #for mi in range(n_modalities):
            #    label_hist[mi, x0:x1, y0:y1, z0:z1] += \
            #            patch[r+dx0:r+dx1, r+dy0:r+dy1, r+dz0:r+dz1] * probs[mi]
            probs = probs[:, np.newaxis, np.newaxis, np.newaxis]
            label_hist[:, x0:x1, y0:y1, z0:z1] += patch[0, r+dx0:r+dx1, r+dy0:r+dy1, r+dz0:r+dz1] * probs
        for i in range(pcoords.shape[0]):
            coord = pcoords[i,:]
            xlabel[pidxs[i],:] = label_hist[:, coord[0], coord[1], coord[2]]
    print "Extracted (%.2f seconds)." % (time.time()-t0)
    return xlabel

def dice(y, ypred):
    A = set(np.nonzero(y)[0])
    B = set(np.nonzero(ypred)[0])
    score = 2*len(A & B) / float(len(A) + len(B))
    return score

def dice_scores(y, ypred, patient_idxs=None, label='Dice scores:'):
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
    ds_mean = np.mean(ds,0)
    ds_std = np.std(ds,0)
    ds_min = np.min(ds,0)
    ds_max = np.max(ds,0)
    print "\n%s" % label
    print "              \tMean\tStd\tMin\tMax"
    print "Whole tumor:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[0], ds_std[0], ds_min[0], ds_max[0])
    print "Tumor core:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[1], ds_std[1], ds_min[1], ds_max[1])
    print "Active tumor:\t%.4f\t%.4f\t%.4f\t%.4f" % (ds_mean[2], ds_std[2], ds_min[2], ds_max[2])

def train_model(xtr, ytr, n_trees=10):
    # Train classifier
    t0 = time.time()
    model = RandomForestClassifier(n_trees, oob_score=True, verbose=1, n_jobs=3)
    #model = ExtraTreesClassifier(n_trees, verbose=1, n_jobs=4)
    #model = svm.SVC(C=1000)
    model.fit(xtr, ytr)
    print "Training took %.2f seconds" % (time.time()-t0)
    #print "OOB score: %.2f%%" % (model.oob_score_*100)
    print "Feature importances:"
    for i in range(4):
        print model.feature_importances_[i*6:(i+1)*6]
    return model

def predict_and_evaluate(model, xte, yte=None, coord=None, dim=None,
                         pred_fname=None, plot_confmat=False, ret_probs=False,
                         patient_idxs=None, pred_img_fname='slices.png'):

    # Predict and evaluate
    if not ret_probs:
        pred = model.predict(xte)
    else:
        pred_probs = model.predict_proba(xte)
        pred = np.argmax(pred_probs, axis=1)

    if yte is not None and patient_idxs is not None:
        print "\nConfusion matrix:"
        cm = confusion_matrix(yte, pred)
        print cm

        acc = sum(pred==yte) / float(len(pred))
        bl_acc = sum(yte==0) / float(len(pred))
        print "Accuracy:\t%.2f%%" % (acc*100)
        print "Majority vote:\t%.2f%%" % (bl_acc*100)

        dice_scores(yte, pred, patient_idxs=patient_idxs)

        if coord is not None and dim is not None:
            # Plot the first patient
            plot_predictions(coord[:patient_idxs[1]], dim[0], pred[:patient_idxs[1]], yte[:patient_idxs[1]], pred_img_fname)
        if pred_fname is not None:
            save_predictions(coord, dim[0], pred, yte, os.path.join('predictions', 'pred_patient%d.csv' % test_patients[0]))

        if plot_confmat:
            plt.figure()
            plot_confusion_matrix(cm)
            plt.show()

    if ret_probs:
        return pred_probs
    else:
        return pred

def main():
    np.random.seed(133742)

    patients = np.random.permutation(40) + 1
    n_tr_p = 1 # Train patients
    n_de_p = 1 # Development patients
    n_te_p = 1 # Test patients
    assert n_tr_p + n_de_p + n_te_p < len(patients), \
            "Not enough patients available"
    train_patients = patients[:n_tr_p]
    test_patients = patients[n_tr_p:n_tr_p+n_te_p]
    dev_patients = patients[n_tr_p+n_te_p:n_tr_p+n_te_p+n_de_p]

    xtr = np.zeros((0,0))
    ytr = np.zeros(0)
    coordtr = np.zeros((0,3))
    patient_idxs_tr = [0]
    dims_tr = []
    print "Train users:"
    for tr_pat in train_patients:
        x, y, coord, dim = load_patient(tr_pat, n_voxels=None)
        ytr = np.concatenate((ytr, y))
        if xtr.shape[0] == 0:
            xtr = x
        else:
            xtr = np.vstack((xtr, x))
        coordtr = np.vstack((coordtr, coord))
        patient_idxs_tr.append(len(ytr))
        dims_tr.append(dim)

    #plot_all_scatter_matrices(xtr, ytr)

    xte = np.zeros((0,0))
    yte = np.zeros(0)
    coordte = np.zeros((0,3))
    patient_idxs_te = [0]
    dims_te = []
    print "Test users:"
    for te_pat in test_patients:
        x, y, coord, dim = load_patient(te_pat, n_voxels=None)
        yte = np.concatenate((yte, y))
        if xte.shape[0] == 0:
            xte = x
        else:
            xte = np.vstack((xte, x))
        coordte = np.vstack((coordte, coord))
        patient_idxs_te.append(len(yte))
        dims_te.append(dim)
    n_non_zeros = sum(yte > 0)
    print "%.1f%% tumor voxels (total %d)" % (100.0 * n_non_zeros / float(len(yte)), len(yte))

    xde = np.zeros((0,0))
    yde = np.zeros(0)
    coordde = np.zeros((0,3))
    patient_idxs_de = [0]
    dims_de = []
    print "Development users:"
    for de_pat in dev_patients:
        x, y, coord, dim = load_patient(de_pat, n_voxels=None)
        yde = np.concatenate((yde, y))
        if xde.shape[0] == 0:
            xde = x
        else:
            xde = np.vstack((xde, x))
        coordde = np.vstack((coordde, coord))
        patient_idxs_de.append(len(yde))
        dims_de.append(dim)

    model = train_model(xtr, ytr, n_trees=10)
    #pred = predict_and_evaluate(model, xte, yte, plot_pred=False, pred_fname=None, plot_confmat=False, ret_probs=False, patient_idxs=patient_idxs_te)
    pred_probs_de = predict_and_evaluate(model, xde, yde, plot_confmat=False, ret_probs=True, patient_idxs=patient_idxs_de)
    xlabel = extract_label_features(coordde, dims_de, pred_probs_de, patient_idxs_de)
    smoothed_pred = np.argmax(xlabel, axis=1)
    print "Development smoothed dice scores:"
    dice_scores(yde, smoothed_pred, patient_idxs=patient_idxs_de)

    xde2 = np.hstack((xde, xlabel))
    model2 = train_model(xde2, yde, n_trees=10)

    print "\n----------------------------------\n"

    pred_probs_te = predict_and_evaluate(
            model, xte, yte, coord=coordte, dim=dims_te, plot_confmat=False,
            ret_probs=True, patient_idxs=patient_idxs_te, pred_img_fname=os.path.join('plots', 'pat%d_slices_0.png' % test_patients[0]))
    for i in range(5):
        xlabel_te = extract_label_features(coordte, dims_te, pred_probs_te, patient_idxs_te)
        smoothed_pred = np.argmax(xlabel_te, axis=1)
        dice_scores(yte, smoothed_pred, patient_idxs=patient_idxs_te, label='Test smoothed dice scores:')

        xte2 = np.hstack((xte, xlabel_te))
        pred_probs_te = predict_and_evaluate(
                model2, xte2, yte, coord=coordte, dim=dims_te, pred_fname=None,
                plot_confmat=False, ret_probs=True, patient_idxs=patient_idxs_te,
                pred_img_fname=os.path.join('plots', 'pat%d_slices_%d.png' % (test_patients[0], i+1)))

if __name__ == "__main__":
    main()
