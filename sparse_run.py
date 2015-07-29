from sparse_encode import sparse_encode;
from patient_plotting import plot_full_feature_scatter_matrix;
from data_processing import load_patients;
import numpy as np;
from methods import train_RF_model;
import data_processing as dp;
import os;
from sklearn.metrics import confusion_matrix;
from evaluation import dice_scores;
from sklearn.externals import joblib;


def predict_two_stages(xtr,ytr,Test_No,n_trees=30):
	fscores=None;
	ytr1 = np.array(ytr, copy=True);
	ytr1[ytr1>0] = 1;
	model1_fname = None;
	model2_fname = None;
	weights = None;
	model1 = train_RF_model(xtr, ytr1, n_trees=n_trees, sample_weight=weights, fname=model1_fname);

	model2 = train_RF_model(xtr, ytr, n_trees=n_trees, sample_weight=weights, fname=model2_fname);
	test_pats = [10];

	yte = np.zeros(0);
	predte = np.zeros(0);
	predte_no_pp = np.zeros(0);
	patient_idxs_te = [0];
	print "Test users:"

	for te_idx, te_pat in enumerate(test_pats):
		X, y, coord, dim = dp.load_patient(te_pat, n_voxels=None);
		m = X.shape[0];
		n = 1;
		k = 25;
		O = sparse_encode(X,m,n,k);

		pred = model1.predict(O);
		best_radius = 6
		pp_pred = dp.post_process(coord, dim, pred, binary_closing=True, radius=best_radius);
		tumor_idxs = pp_pred > 0;
		pred_probs2 = model2.predict_proba(O[tumor_idxs,:]);
		pred2 = np.argmax(pred_probs2, axis=1) + 1;
		pp_pred[tumor_idxs] = pred2;
		pp_pred15 = np.array(pp_pred);
		print "\nConfusion matrix:";
		cm = confusion_matrix(y, pp_pred15);
		print cm;
		dice_scores(y, pp_pred15, label='Dice scores:')

		# Closing post processing
		pp_pred[tumor_idxs] = dp.post_process(coord[tumor_idxs,:], dim, pred2, remove_components=False, radius=best_radius);
		method = 'closing'


		print "\nConfusion matrix (pp):";
		cm = confusion_matrix(y, pp_pred);
		print cm;

		yte = np.concatenate((yte, y));
		patient_idxs_te.append(len(yte));
		predte = np.concatenate((predte, pp_pred));
		predte_no_pp = np.concatenate((predte_no_pp, pp_pred15))


		dice_scores(y, pp_pred, label='Dice scores (pp):')



	print "\nOverall confusion matrix:"
	cm = confusion_matrix(yte, predte);
	print cm


	dice_scores(yte, predte_no_pp, patient_idxs=patient_idxs_te,label='Overall dice scores (two-stage, no pp):', fscores=fscores);

	dice_scores(yte, predte, patient_idxs=patient_idxs_te,label='Overall dice scores (two-stage):', fscores=fscores);


if __name__ == '__main__':
	
	#pat_ind = [1,4,10,12,13,14,23,25,28,33,36];
	pat_ind = [1];

	xtr, ytr, coordtr, patient_idxs_tr, dims_tr = load_patients(pat_ind);
	idx = ytr > 0
	Y = ytr[idx]
	X = xtr[idx,:]
	#Y[Y < 4]=0;
	#Y[Y == 4]=1;
	print X.shape;
	m = X.shape[0];
	n = 1;
	k = 25;
	O = sparse_encode(X,m,n,k);
	print O.shape;
	print Y.shape
	predict_two_stages(O,Y,[10]);
