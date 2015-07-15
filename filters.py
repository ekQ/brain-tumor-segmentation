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
	return S+S.T;
	

def skimage_apply_filters(mat,connectivity=2):	#expects numpy array
	selem = ball(10);
	mat = closing(mat,selem);

	label_mat,rc = label(mat,background=0,return_num=True,connectivity=2);
	label_mat+=1;	#background has to be labelled as 0, not as -1;
	rc +=1;	#size is 1 + maximum label.
	index_by_value = collections.defaultdict(dict);
	cnt = np.zeros(rc);
	for i,x in np.ndenumerate(label_mat):
		if 'r' in index_by_value[x]:
			index_by_value[x]['r'].append(int(i[0]));
		else:
			index_by_value[x]['r'] = [int(i[0])];
		if 'c' in index_by_value[x]:
			index_by_value[x]['c'].append(int(i[1]));
		else:
			index_by_value[x]['c'] = [int(i[1])];
		cnt[x] += 1;

	#Get rid of small connected components
	for j in np.nditer(np.where(cnt<10)):	#get indices whose count smaller than 20;
		t = index_by_value[int(j)];
		mat[t['r'],t['c']]=0;	#set it to background;

	
	#selem = ball(10);
	#footprint = sp.ndimage.generate_binary_structure(3,1);
	#eroded = sp.ndimage.grey_erosion(mat, size=(3,3,3));
	#eroded = erosion(mat, selem);
	#return opening(mat,selem);
	#mat = opening(mat,selem);
	#return label(mat,background=0,connectivity=2);
	
	#rgb_mat = color.lab2rgb(label_mat);
	#return color.rgb2grey(rgb_mat);
	return mat;


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
