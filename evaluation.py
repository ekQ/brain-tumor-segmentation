import numpy as np

def dice(y, ypred):
    A = set(np.nonzero(y)[0])
    B = set(np.nonzero(ypred)[0])
    if len(A) == 0:
        return 1 # Not sure how it should be computed in this case
    score = 2*len(A & B) / max(float(len(A) + len(B)), 1)
    return score

def dice_scores(y, ypred, patient_idxs=None, label='Dice scores:', fscores=None):
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
    scores_str = ""
    scores_str += "\n%s\n" % label
    scores_str += "             \tMean\tStd\tMin\tMax\n"
    scores_str += "Whole tumor: \t%.4f\t%.4f\t%.4f\t%.4f\n" % (ds_mean[0], ds_std[0], ds_min[0], ds_max[0])
    scores_str += "Tumor core:  \t%.4f\t%.4f\t%.4f\t%.4f\n" % (ds_mean[1], ds_std[1], ds_min[1], ds_max[1])
    scores_str += "Active tumor:\t%.4f\t%.4f\t%.4f\t%.4f\n" % (ds_mean[2], ds_std[2], ds_min[2], ds_max[2])
    print scores_str
    if fscores is not None:
        fscores.write(scores_str)
    return ds_mean
