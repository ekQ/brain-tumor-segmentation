import skimage.morphology
import scipy.io
import time
import numpy as np
import os
import sys
from pystruct.inference import inference_dispatch, compute_energy

from evaluation import dice_scores

def preprocess(x):
    # Median to zero
    x -= np.median(x,0)
    # Variance to 1
    x /= np.std(x,0)
    return x

def post_process(coord, dim, pred, pred_probs=None, remove_components=True,
                 binary_closing=False, radius=6):
    t0 = time.time()
    # 3D data matrix
    D = np.ones((dim[0], dim[1], dim[2]), dtype=int) * -1
    for i in range(coord.shape[0]):
        D[coord[i,0], coord[i,1], coord[i,2]] = pred[i]
    
    neighborhood = skimage.morphology.ball(radius)
    
    if binary_closing:
        D2 = D > 0
        D = skimage.morphology.binary_closing(D2, neighborhood)
        #D[D3==0] = 0
        #D[np.logical_and(D==0, D3==1)] = 2
    else:
        D = skimage.morphology.closing(D, neighborhood)
    
    if remove_components:
        remove_small_components(D)

    new_pred = []
    for i in range(coord.shape[0]):
        new_pred.append(D[coord[i,0], coord[i,1], coord[i,2]])
    print "Post-processing took %.2f seconds." % (time.time()-t0)
    return np.array(new_pred, dtype=int)

def create_graph(coords):
    n = coords.shape[0]
    coords = coords.astype(np.int32)

    t0 = time.time()
    neighs = []
    for i in range(-1,1):
        for j in range(-1,1):
            for k in range(-1,1):
                if i==0 and j==0 and k==0:
                    continue
                neighs.append(np.array([i,j,k],dtype=np.int32))
    vox_map = {}
    edges = []
    for l in range(n):
        coord = coords[l,:]
        coord_tup = tuple(coord)
        assert coord_tup not in vox_map, "Same coordinate appearing twice %s" % str(coord_tup)
        coord_idx = len(vox_map)
        vox_map[coord_tup] = coord_idx
        for neigh in neighs:
            coord2 = coord + neigh
            coord2_tup = tuple(coord2)
            if coord2_tup in vox_map:
                edges.append((coord_idx, vox_map[coord2_tup]))
    print "Graph creation took %.2f seconds (%d edges)." % (time.time()-t0, len(edges))
    edges = np.asarray(edges, dtype=np.int32)
    return edges

def mrf(probs, edges, potential=None):
    #probs2 = (-100 * np.log(probs)).astype(np.int32)
    #probs2 = (100 * probs).astype(np.int32)
    #min_prob = 0.001
    probs2 = np.array(probs)
    #probs2[probs2 < min_prob] = min_prob
    #probs2 = np.log(probs2)
    if potential is None:
        n_labels = probs2.shape[1]
        potential = np.eye(n_labels, dtype=np.int32)
    print "%d labels." % probs2.shape[1]
    t0 = time.time()
    smoothed_pred = inference_dispatch(probs2, potential, edges,
                                       inference_method='qpbo')
    print "MRF took %.2f seconds." % (time.time()-t0)
    return smoothed_pred

def remove_small_components(D, min_component_size=3000):
    t0 = time.time()
    C, n_components = scipy.ndimage.measurements.label(D)
    n_removed = 0
    for i in range(1,n_components+1):
        component = np.nonzero(C==i)
        if len(component[0]) < min_component_size:
            D[component] = 0
            n_removed += 1
    print "Removed %d out of %d components (%.2f seconds)." % (n_removed, n_components, time.time()-t0)

def post_process_multi_radii(coord, dim, pred, radii, y=None,
                             remove_components=True, binary_closing=False):
    t0 = time.time()
    # 3D data matrix
    D_orig = np.ones((dim[0], dim[1], dim[2]), dtype=int) * -1
    for i in range(coord.shape[0]):
        D_orig[coord[i,0], coord[i,1], coord[i,2]] = pred[i]

    all_preds = []
    for r in radii:
        D = np.array(D_orig) # Copy array
        neighborhood = skimage.morphology.ball(r)

        if binary_closing:
            D2 = D > 0
            D = skimage.morphology.binary_closing(D2, neighborhood)
            #D[D3==0] = 0
            #D[np.logical_and(D==0, D3==1)] = 2
        else:
            D = skimage.morphology.closing(D, neighborhood)

        if remove_components:
            remove_small_components(D)

        new_pred = []
        for i in range(coord.shape[0]):
            new_pred.append(D[coord[i,0], coord[i,1], coord[i,2]])
        all_preds.append(new_pred)
        # Evaluation
        if y is not None:
            dice_scores(y, new_pred, patient_idxs=None,
                        label='Dice scores (r=%d):' % r)

    ret = np.zeros((len(all_preds[0]), len(all_preds)))
    for i, pred in enumerate(all_preds):
        ret[:,i] = pred

    print "Post-processing took %.2f seconds." % (time.time()-t0)
    return np.asarray(pred, dtype=int)

class_counts = np.zeros(5)

def load_patient(number, do_preprocess=True, n_voxels=None, stratified=False):
    data = scipy.io.loadmat(os.path.join('data', 'Patient_Features_%d.mat' % number))
    data = data['featuresMatrix']

    tumor_grade = data[0,0]
    print "Patient %d, tumor grade: %d" % (number, tumor_grade)

    row0 = 5
    y = data[row0:, 1]
    x = data[row0:, 5:]
    print "Features available: %d" % x.shape[1]
    #x = x[:, [19, 18, 10, 0, 79, 9, 70, 69, 8, 15, 60]]
    #x = data[row0:, 5:11]
    #x = data[row0:, [5,11,17,23]]
    if do_preprocess:
        x = preprocess(x)
        pass

    # Update class counts
    new_counts = np.histogram(y, bins=range(6))[0]
    global class_counts
    class_counts += new_counts

    coord = data[row0:, 2:5]
    if n_voxels is not None and isinstance(n_voxels, int):
        idxs = np.random.permutation(len(y))
        if not stratified:
            idxs = idxs[:min(n_voxels,len(y))]
            y = y[idxs]
            x = x[idxs,:]
            coord = coord[idxs,:]
        else:
            y = y[idxs]
            x = x[idxs,:]
            coord = coord[idxs,:]
            x2 = np.zeros((0,x.shape[1]), dtype=np.float32)
            y2 = np.zeros(0)
            coord2 = np.zeros((0,3))
            n_batch = int(n_voxels / 8)
            for i in range(5):
                if i == 0:
                    new_idxs = np.nonzero(y==i)[0][:4*n_batch]
                else:
                    new_idxs = np.nonzero(y==i)[0][:n_batch]
                x2 = np.vstack((x2, x[new_idxs,:]))
                y2 = np.concatenate((y2, y[new_idxs]))
                coord2 = np.vstack((coord2, coord[new_idxs]))
            x = x2
            y = y2
            coord = coord2

    dim = data[3, :3]

    # Make sure data type is float32 as it might be more memory efficient sklearn.fit
    x = np.asarray(x, dtype=np.float32)

    # Remove bad values
    print "Max:", x.max(), " Min:", x.min()
    x[np.isnan(x)] = 0

    return x, y, coord, dim

def load_patients(pats, stratified=False):
    xtr = np.zeros((0,0), dtype=np.float32)
    ytr = np.zeros(0)
    coordtr = np.zeros((0,3))
    patient_idxs_tr = [0]
    dims_tr = []
    for pat in pats:
        x, y, coord, dim = load_patient(pat, n_voxels=30000,
                                        stratified=stratified)
        ytr = np.concatenate((ytr, y))
        if xtr.shape[0] == 0:
            xtr = x
        else:
            xtr = np.vstack((xtr, x))
        coordtr = np.vstack((coordtr, coord))
        patient_idxs_tr.append(len(ytr))
        dims_tr.append(dim)
    return xtr, ytr, coordtr, patient_idxs_tr, dims_tr

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
