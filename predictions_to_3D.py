# -*- coding: utf-8 -*-
"""
Transform 3D brain tumor predictions to a 3D image.
"""
import numpy as np
import cPickle as pickle
import os.path

def predictions_to_3d(pred_fname, fpickle=None):
    """
    Input:
        pred_fname  Path to the file which contains the prediction. Note that
                    pred_fname+'.dim' must also exist.
        fpickle     Store the 3D array to this path, unless fpickle is None
    """
    # Load predictions    
    X = np.loadtxt(open(pred_fname,"rb"), delimiter=",", skiprows=1)
    coord = X[:,:3]
    pred = X[:,3]
    orig = X[:,4];

    # Load dimensions of the data
    fdim = open(pred_fname+'.dim', 'r')
    dim = map(int, fdim.readline().strip().split(','))
    print "Matrix dimensions are:", dim

    # 3D data matrix
    D = np.ones((dim[0], dim[1], dim[2])) * -1
    E = np.ones((dim[0], dim[1], dim[2])) * -1
    for i in range(coord.shape[0]):
        D[coord[i,0], coord[i,1], coord[i,2]] = pred[i]
        E[coord[i,0], coord[i,1], coord[i,2]] = orig[i]
    
    # Save the prediction matrix to file if filename is provided
    if fpickle is not None:
        with open(fpickle, 'wb') as fp:
            pickle.dump(D, fp)
    
    return (D,E);
    
if __name__ == "__main__":
    patient_id = 174
    pred_fname = os.path.join('predictions', 'pat%d_pred.csv' % patient_id)
    D = predictions_to_3d(pred_fname, os.path.join('predictions', 'pat%d_pred.pckl' % patient_id))
    print "Transformed a matrix of size:", D.shape
