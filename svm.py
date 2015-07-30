from sklearn import svm;
from sklearn.metrics import jaccard_similarity_score,confusion_matrix;
import data_processing as dp;
import numpy as np;

def learn_model(X,Y,c,g):
	clf = svm.SVC(C=2**c,gamma=2**g,kernel='rbf',probability=False);
	clf.fit(X,Y);
	return clf;
	

def split_train_val_test(X,Y):
	np.random.seed(1000);
	rand_perm = np.random.permutation(Y.shape[0]);
	tr_sz = Y.shape[0]/2;	#Half of the data for training
	vc_sz = Y.shape[0]/4;	#Quarter data for validation
	tr_ind = rand_perm[0:tr_sz];
	vc_ind = rand_perm[tr_sz:tr_sz+vc_sz];
	te_ind = rand_perm[tr_sz+vc_sz:]
	return X[tr_ind,:],Y[tr_ind],X[vc_ind,:],Y[vc_ind],X[te_ind,:],Y[te_ind];


def select_rbf_hyper_params(Xtr,Ytr,Xvc,Yvc,C,G):
	Dice = 0;
	c_b = g_b = -1;
	model = None;
	
	for c_b in C:
		for g_b in G:
			b_model = learn_model(Xtr,Ytr,c_b,g_b);
			pred = b_model.predict(Xvc);
			J = jaccard_similarity_score(Yvc,pred);
			b_Dice = 2*J/float(1+J);
			print "C: %f G: %f Dice: %f" %(c_b,g_b,b_Dice);
			if b_Dice > Dice:
				Dice = b_Dice;
				c = c_b;
				g = g_b;
				model = b_model;

			
	return model,c,g,Dice;


if __name__ == '__main__':
	
	pat_ind = [1,4,10,12,13,14,23,25,28,33,36];
	X,Y = dp.load_patients(pat_ind)[0:2];
	idx = Y > 0;
	X = X[idx,:];
	Y = Y[idx];
	np.random.seed(1000);
	rand_idx = np.random.permutation(Y.shape[0])[0:10000];
	Y = Y[rand_idx];
	X = X[rand_idx,:];
	Xtr,Ytr,Xvc,Yvc,Xte,Yte = split_train_val_test(X,Y);
	print sum(Ytr == 1);
	print sum(Ytr == 2);
	print sum(Ytr == 3);
	print sum(Ytr == 4); 
	C = xrange(-4,4);
	G = xrange(-4,4);
	model,c,g,Dice = select_rbf_hyper_params(Xtr,Ytr,Xvc,Yvc,C,G);
	pred = model.predict(Xte);
	J = jaccard_similarity_score(Yte,pred);
	Dice = 2*J/float(1+J);
	print "Test Dice: %f" %Dice;
	print confusion_matrix(Yte,pred);
	

	
	


