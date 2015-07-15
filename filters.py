from skimage.measure import label;
import skimage.color as color;
from predictions_to_3D import predictions_to_3d;
from evaluation import dice_scores;
import numpy as np;
import scipy as sp;
import SimpleITK as sitk;
import scipy.io as sio
from skimage.morphology import erosion, dilation, opening, closing, disk, ball;
import collections;


def gen_synth_data(row=4,col=4):
	mdiag1 = [1]*(col-1);
	mdiag1.append(0);
	minor_diag1 = np.tile(mdiag1,(1,row)).flatten();
	minor_diag1 = np.delete(minor_diag1,-1);
	
	mdiag2 = [1]*col*(row-1)
	minor_diag2 = np.array(mdiag2);

	S = np.diag(minor_diag1,1) + np.diag(minor_diag2,col);
	print S+S.T;
	return S+S.T;
	

def skimage_apply_filters(mat,connectivity=2):	#expects numpy array
	label_mat = label(mat,background=0,connectivity=1);
	rc = np.max(label_mat);
	index_by_value = collecions.defaultdict(list);
	cnt = np.zeors(rc);
	for i,x in np.ndenumerae(label_mat):
		index_by_value[x].append(i);
		cnt[x] += 1;
	
	for in
	
	
	selem = ball(10);
	#footprint = sp.ndimage.generate_binary_structure(3,1);
	#eroded = sp.ndimage.grey_erosion(mat, size=(3,3,3));
	#eroded = erosion(mat, selem);
	#return opening(mat,selem);
	return closing(mat,selem);
	#return label(mat,background=0,connectivity=2);
	
	#rgb_mat = color.lab2rgb(label_mat);
	#return color.rgb2grey(rgb_mat);


def store_as_csv(mat):
	np.savetxt("conn_comp.csv",mat,delimiter=',',fmt="%.1u");


def call_dice(L_M,E):	#For testing purpose
	(l,m,n) = E.shape;
	y = list();
	ypred = list();

	for i in xrange(l):
		for j in xrange(m):
			for k in xrange(n):
				if ((E[i,j,k] == -1) and (L_M[i,j,k] == -1)):
					continue;
				else:
					y.append(int(E[i,j,k]));
					ypred.append(int(L_M[i,j,k]));
	dice_scores(np.array(y),np.array(ypred));
	
	

def vigra_apply_conn_comp(mat,connectivity=4):	#expects numpy array
	#vmat = mat.transposeToVigraOrder();	#numpy and Vigra uses different memory layout
	vmat = mat.T;
	#ImObj = vigra.impex.readImage(Ifile);
	label_vmat = vigra.analysis.labelImageWithBackground(vmat,neighborhood = connectivity);
	vmat = label_vmat.transposeToNumpyOrder();
	return vmat;

def export_to_MHA(data):	#input numpy array
	img = sitk.GetImageFromArray(data);
	sitk.WriteImage(img,"data/ProcessedImage.mha");


def imread(fname):
	sitk_img = sitk.ReadImage(fname);
	return sitk.GetArrayFromImage(sitk_img);	#Get Numpy array from the image

if __name__ == "__main__":
	#S=gen_synth_data(row=2,col=2);
	#print skimage_apply_conn_comp(S);
	pred_file = 'predictions/pat174_pred.csv';
	(M,E) = predictions_to_3d(pred_file);
	#sio.savemat('original.mat',{'Orig':E});
	#sio.savemat('predict.mat',{'Pred':M});
	#export_to_MHA(skimage_apply_conn_comp(M));
	M[M<0]=0;	#Prediction
	E[E<0]=0;	#Original
	L_M = skimage_apply_filters(M);	#New prediction
	call_dice(L_M,E);
	
	#print L_M[L_M>0];
	#print L_M.shape;
	#L_M = vigra_apply_conn_comp(M);
	#call_dice(E,L_M);
