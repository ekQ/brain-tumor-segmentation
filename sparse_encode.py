from skimage import data;
import spams;
import numpy as np;
from PIL import Image
from pylab import *

X1 = np.random.randint(255,size=(128,128));
X2 = np.random.randint(255,size=(128,128));
X3 = np.random.randint(255,size=(128,128));

X = np.hstack((X1,X2,X1,X3,X2,X3));
imshow(X,cmap="Greys_r");
show();
#X = data.camera();
#X = X/255. ;
mx = np.max(np.max(X,axis=0));
X = X/float(mx);
#print X.ndim;
if X.ndim == 3:
	A = np.asfortranarray(X.reshape((X.shape[0],X.shape[1]*X.shape[2])));
	rgb = True;
else:
	A = np.asfortranarray(X);
	rgb = False;
#print A.shape;
m = 128;	#No of rows in the image
n = 128;	#No of cols in the image
#A new matrix with 512*512 rows is created. One image corresponds to one column. In fact we do sliding img2col, so if we want we can split the image into small blacks.
X = spams.im2col_sliding(A,m,n,rgb);
X = X - np.tile(np.mean(X,0),(X.shape[0],1));
X = np.asfortranarray(X / np.tile(np.sqrt((X * X).sum(axis=0)),(X.shape[0],1)),dtype = float);

#print X.shape;
K = 3;
param = { 'K' : K, 'mode' : 0, 'modeD' : 0, 'lambda1' : 0.15, 'numThreads' : 4, 'batchsize' : 400, 'iter' : 50};

D = spams.trainDL(X,**param);
#print D.shape;
O=np.zeros((m,n*K),dtype=float);
for i in range(D.shape[1]):
	O[0:m,i*n:i*n+n] = reshape(D[:,i],(m,n),order='F');

#print mx*D;
imshow(O,cmap="Greys_r");
show();
#X = data.camera();
