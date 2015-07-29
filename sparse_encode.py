from sklearn import datasets;
from skimage import data;
import spams;
import numpy as np;
from PIL import Image
from pylab import *

def iris_test():
	iris = datasets.load_iris();
	X=iris.data;
	#X1 = np.random.randint(255,size=(128,128));
	#X2 = np.random.randint(255,size=(128,128));
	#X3 = np.random.randint(255,size=(128,128));
	#X = np.hstack((X1,X2,X1,X3,X2,X3));
	#imshow(X,cmap="Greys_r");
	#show();
	#X = data.camera();
	#X = X/255. ;
	#In normal settings (multile images), m*n is the total number of features of the image. for eg: 256*256 image has that much no of features. Here our aim is to learn subsample. 
	m = 150;	#In our case no of pixels.
	n = 1;	#one features are taken together to form one sample. (totally 4 features present for iris data).
	k = 2;	#no of features required in the output dictionaries. 
	sparse_encode(X,m,n,k);

def sparse_encode(X,m,n,k):	#Original Data, No of Rows, No of Columns, Output Columns Required

	mx = np.max(np.max(X,axis=0));
	X = X/float(mx);	#Normalize
	if X.ndim == 3:
		A = np.asfortranarray(X.reshape((X.shape[0],X.shape[1]*X.shape[2])));
		rgb = True;
	else:
		A = np.asfortranarray(X);
		rgb = False;
	#A new matrix with m*n rows and (R-m+1)*(C-n+1) R = no of row of X and C = no of col of X is created. One image(patch) corresponds to one column in the resulting matrix..
	X = spams.im2col_sliding(A,m,n,rgb);
	X = X - np.tile(np.mean(X,0),(X.shape[0],1));
	X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)),dtype = float);

	param = { 'K' : k, 'mode' : 0, 'modeD' : 0, 'lambda1' : 0.15, 'numThreads' : 4, 'batchsize' : 400, 'iter' : 100};

	D = spams.trainDL(X,**param);
	print D.shape;
	O=np.zeros((m,n*k),dtype=float);
	for i in range(D.shape[1]):
		O[0:m,i*n:i*n+n] = reshape(D[:,i],(m,n),order='F');

	return O;


if __name__ == '__main__':
	iris_test();	
