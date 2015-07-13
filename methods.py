from evaluation import dice_scores
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import time
import os

import patient_plotting as pp
import extras
import data_processing as dp

def predict_RF(train_pats, test_pats, fscores=None, plot_predictions=False,
               stratified=False):
    """
    Predict tumor voxels for given test patients.

    Input:
        train_pats -- list of patient IDs used for training a model.
        test_pats -- list of patient IDs used for testing a model.
        fscores -- An opened output file to which we write the results.
    """
    xtr, ytr, coordtr, patient_idxs_tr, dims_tr = dp.load_patients(train_pats,
                                                                   stratified)

    # Class frequencies in the whole dataset
    class_freqs = dp.class_counts / float(sum(dp.class_counts))
    print "Class frequencies:", class_freqs*100
    # Class frequencies in the sample
    sample_counts = np.histogram(ytr, range(6))[0]
    sample_freqs = sample_counts / float(sum(sample_counts))
    print "Sample frequencies:", sample_freqs*100
    weights = np.ones(len(ytr))
    for i in range(5):
        weights[ytr==i] = class_freqs[i] / sample_freqs[i]
    #weights = None
    model = train_RF_model(xtr, ytr, n_trees=30, sample_weight=weights)

    print "\n----------------------------------\n"

    yte = np.zeros(0)
    predte = np.zeros(0)
    patient_idxs_te = [0]
    print "Test users:"
    # Iterate over test users
    for te_idx, te_pat in enumerate(test_pats):
        print "Test patient number %d" % (te_idx+1)
        x, y, coord, dim = dp.load_patient(te_pat, n_voxels=None)

        if plot_predictions:
            pif = os.path.join('plots', 'pat%d_slices_0_RF.png' % te_pat)
        else:
            pif = None
        pred_probs_te = predict_and_evaluate(
                model, x, y, coord=coord, dim_list=[dim], plot_confmat=False,
                ret_probs=True, patient_idxs=None,
                pred_img_fname=pif)
        pred = np.argmax(pred_probs_te, axis=1)

        yte = np.concatenate((yte, y))
        patient_idxs_te.append(len(yte))
        predte = np.concatenate((predte, pred))
        '''
        for i in range(1):
            xlabel_te = dp.extract_label_features(coordte, dims_te, pred_probs_te,
                                                  patient_idxs_te)
            smoothed_pred = np.argmax(xlabel_te, axis=1)
            dice_scores(yte, smoothed_pred, patient_idxs=patient_idxs_te,
                        label='Test smoothed dice scores (iteration %d):' % (i+1))
    
            xte2 = np.hstack((xte, xlabel_te))
            pred_probs_te = predict_and_evaluate(
                    model2, xte2, yte, coord=coordte, dim_list=dims_te, pred_fname=None,
                    plot_confmat=False, ret_probs=True, patient_idxs=patient_idxs_te,
                    pred_img_fname=os.path.join('plots', 'pat%d_slices_%d.png' % (test_pats[0], i+1)))
        '''

    print "\nOverall confusion matrix:"
    cm = confusion_matrix(yte, predte)
    print cm

    dice_scores(yte, predte, patient_idxs=patient_idxs_te,
                label='Overall dice scores (RF):', fscores=fscores)

def predict_two_stage(train_pats, test_pats, fscores=None,
                      plot_predictions=False, stratified=False):
    """
    Predict tumor voxels for given test patients.

    Input:
        train_pats -- list of patient IDs used for training a model.
        test_pats -- list of patient IDs used for testing a model.
        fscores -- An opened output file to which we write the results.
    """
    xtr, ytr, coordtr, patient_idxs_tr, dims_tr = dp.load_patients(train_pats,
                                                                   stratified)

    # Make all tumor labels equal to 1 and train the first model
    ytr1 = np.array(ytr, copy=True)
    ytr1[ytr1>0] = 1
    if stratified:
        # Class frequencies in the whole dataset
        class_counts = [dp.class_counts[0], sum(dp.class_counts[1:])]
        class_freqs = np.asarray(class_counts) / float(sum(class_counts))
        print "Class frequencies (model 1):", class_freqs*100
        # Class frequencies in the sample
        sample_counts = np.histogram(ytr, [0,1,5])[0]
        sample_freqs = sample_counts / float(sum(sample_counts))
        print "Sample frequencies:", sample_freqs*100
        weights = np.ones(len(ytr))
        for i in range(2):
            weights[ytr==i] = class_freqs[i] / sample_freqs[i]
    else:
        weights = None
    model1 = train_RF_model(xtr, ytr1, n_trees=30, sample_weight=weights)

    # Train the second model to separate tumor classes
    ok_idxs = ytr > 0
    xtr2 = np.asarray(xtr[ok_idxs,:])
    ytr2 = np.asarray(ytr[ok_idxs])
    if stratified:
        # Class frequencies in the whole dataset
        class_counts = dp.class_counts[1:]
        class_freqs = np.asarray(class_counts) / float(sum(class_counts))
        print "Class frequencies (model 2):", class_freqs*100
        # Class frequencies in the sample
        sample_counts = np.histogram(ytr, range(1,6))[0]
        sample_freqs = sample_counts / float(sum(sample_counts))
        print "Sample frequencies:", sample_freqs*100
        weights = np.ones(len(ytr2))
        for i in range(4):
            weights[ytr2==i+1] = class_freqs[i] / sample_freqs[i]
    else:
        weights = None
    model2 = train_RF_model(xtr2, ytr2, n_trees=30, sample_weight=weights)

    print "\n----------------------------------\n"

    yte = np.zeros(0)
    predte = np.zeros(0)
    patient_idxs_te = [0]
    print "Test users:"
    # Iterate over test users
    for te_idx, te_pat in enumerate(test_pats):
        print "Test patient number %d" % (te_idx+1)
        x, y, coord, dim = dp.load_patient(te_pat, n_voxels=None)

        pred = model1.predict(x)
        pp_pred = dp.post_process(coord, dim, pred)

        tumor_idxs = pp_pred == 1
        pred2 = model2.predict(x[tumor_idxs,:])
        pp_pred[tumor_idxs] = pred2

        print "\nConfusion matrix:"
        cm = confusion_matrix(y, pp_pred)
        print cm

        yte = np.concatenate((yte, y))
        patient_idxs_te.append(len(yte))
        predte = np.concatenate((predte, pp_pred))

        dice_scores(y, pp_pred, label='Dice scores (pp):')

        if plot_predictions:
            # Plot the patient
            pif = os.path.join('results', 'pat%d_slices_0_2S.png' % te_pat)
            pp.plot_predictions(coord, dim, pp_pred, y, fname=pif)
            #if pred_fname is not None:
            #    extras.save_predictions(coord, dim_list[0], pred, yte, pred_fname)

    print "\nOverall confusion matrix:"
    cm = confusion_matrix(yte, predte)
    print cm

    dice_scores(yte, predte, patient_idxs=patient_idxs_te,
                label='Overall dice scores (two-stage):', fscores=fscores)


def train_RF_model(xtr, ytr, n_trees=10, sample_weight=None):
    # Train classifier
    t0 = time.time()
    model = RandomForestClassifier(n_trees, oob_score=True, verbose=1, n_jobs=3)
    #model = ExtraTreesClassifier(n_trees, verbose=1, n_jobs=4)
    #model = svm.SVC(C=1000)
    model.fit(xtr, ytr, sample_weight=sample_weight)
    print "Training took %.2f seconds" % (time.time()-t0)
    #print "OOB score: %.2f%%" % (model.oob_score_*100)
    print "Feature importances:"
    for i in range(4):
        print model.feature_importances_[i*20:(i+1)*20]
    return model


def predict_and_evaluate(model, xte, yte=None, coord=None, dim_list=None,
                         pred_fname=None, plot_confmat=False, ret_probs=False,
                         patient_idxs=None, pred_img_fname=None,
                         pred_3D_fname=None):

    # Predict and evaluate
    if not ret_probs:
        pred = model.predict(xte)
    else:
        pred_probs = model.predict_proba(xte)
        pred = np.argmax(pred_probs, axis=1)

    if yte is not None:
        print "\nConfusion matrix:"
        cm = confusion_matrix(yte, pred)
        print cm

        acc = sum(pred==yte) / float(len(pred))
        bl_acc = sum(yte==0) / float(len(pred))
        print "Accuracy:\t%.2f%%" % (acc*100)
        print "Majority vote:\t%.2f%%" % (bl_acc*100)

        dice_scores(yte, pred, patient_idxs=patient_idxs)
        
        pp_pred = dp.post_process(coord, dim_list[0], pred)
        dice_scores(yte, pp_pred, patient_idxs=patient_idxs, label='Dice scores (pp):')

        if coord is not None and dim_list is not None and pred_img_fname is not None:
            # Plot the first patient
            if patient_idxs is None:
                patient_idxs = [0, len(yte)]
            pp.plot_predictions(coord[:patient_idxs[1]], dim_list[0], 
                                pred[:patient_idxs[1]], yte[:patient_idxs[1]],
                                pp_pred=pp_pred, fname=pred_img_fname,
                                fpickle=pred_3D_fname)
        if pred_fname is not None:
            extras.save_predictions(coord, dim_list[0], pred, yte, pred_fname)

        if plot_confmat:
            plt.figure()
            pp.plot_confusion_matrix(cm)
            plt.show()

    if ret_probs:
        return pred_probs
    else:
        return pp_pred
