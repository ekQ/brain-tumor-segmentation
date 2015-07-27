import numpy as np
import time
import datetime as dt
import os
import re
import random
import sys

import patient_plotting as pp
import extras
import methods

# Experiment parameters
seed = 98234111
n_tr_p = 100 # Train patients
n_de_p = 0 # Development patients
n_te_p = 100 # Test patients
stdout2file = True
n_trees = 30
plot_predictions = False
stratified = False
resolution = 2 # 1/2/4, 1 is the highest, 2 is 2^3 times smaller

def run_experiment(method):
    # Plot parameters to store them to output log
    print "seed", seed
    print "n_tr_p", n_tr_p
    print "n_te_p", n_te_p
    print "n_de_p", n_de_p
    print "n_trees", n_trees
    print "resolution", resolution

    method_names = {1:'RF', 2:'two-stage', 3:'online'}
    datestr = re.sub('[ :]','',str(dt.datetime.now())[:-7])
    res_dir = datestr + '_' + method_names[method]
    os.makedirs(os.path.join('results', res_dir))
    global fscores
    fscores = open(os.path.join('results', res_dir, "results_%s_seed%d.txt" % (datestr,seed)), 'w')
    
    np.random.seed(seed)
    random.seed(seed)

    t_beg = time.time()
    if resolution == 1:
        pat_fname = "Patient_Features_(\d+)\.mat"
    elif resolution == 2:
        pat_fname = "Patient_Features_SubsampleX2_(\d+)\.mat"
    elif resolution == 4:
        pat_fname = "Patient_Features_SubsampleX4_(\d+)\.mat"
    else:
        raise ValueError('Resolution must be 1, 2, or 4')
    available_files = os.listdir('data')
    patients = []
    for f in available_files:
        m = re.match(pat_fname, f)
        if m:
            patients.append(int(m.group(1)))
    random.shuffle(patients)
    print patients
    assert n_tr_p + n_de_p + n_te_p < len(patients), \
            "Not enough patients available"
    test_patients = patients[:n_te_p]
    train_patients = patients[n_te_p:n_te_p+n_tr_p]
    dev_patients = patients[n_te_p+n_tr_p:n_te_p+n_tr_p+n_de_p]

    if method == 1:
        methods.predict_RF(train_patients, test_patients, fscores,
                           plot_predictions, stratified)
    elif method == 2:
        methods.predict_two_stage(train_patients, test_patients, fscores,
                                  plot_predictions, stratified, n_trees,
                                  dev_pats=dev_patients, use_mrf=False,
                                  resolution=resolution)
    elif method == 3:
        methods.predict_online(train_patients, test_patients, fscores,
                               plot_predictions)
    else:
        print "Unknown method:", method

    print "Total time: %.2f seconds." % (time.time()-t_beg)
    fscores.close()

def main():
    run_experiment(2)
    #run_experiment(1)

if __name__ == "__main__":
    datestr = re.sub('[ :]','',str(dt.datetime.now())[:-7])
    if stdout2file:
        sys.stdout = open(os.path.join('results', "stdout_%s_seed%d_ntrp%d_ntep%d_res%d.txt" % (datestr,seed, n_tr_p, n_te_p, resolution)), 'w')
        sys.stderr = open(os.path.join('results', "stderr_%s_seed%d_ntrp%d_ntep%d_res%d.txt" % (datestr,seed, n_tr_p, n_te_p, resolution)), 'w')
    main()
