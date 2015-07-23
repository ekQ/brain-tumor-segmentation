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
n_tr_p = 2 # Train patients
n_de_p = 1 # Development patients
n_te_p = 1 # Test patients
stdout2file = False
plot_predictions = True
stratified = False

def run_experiment(method):

    method_names = {1:'RF', 2:'two-stage', 3:'online'}
    datestr = re.sub('[ :]','',str(dt.datetime.now())[:-7])
    res_dir = datestr + '_' + method_names[method]
    os.makedirs(os.path.join('results', res_dir))
    global fscores
    fscores = open(os.path.join('results', res_dir, "results_%s_seed%d.txt" % (datestr,seed)), 'w')
    
    np.random.seed(seed)
    random.seed(seed)

    t_beg = time.time()
    available_files = os.listdir('data')
    patients = []
    for f in available_files:
        m = re.match("Patient_Features_(\d+)\.mat", f)
        if m:
            patients.append(int(m.group(1)))
    random.shuffle(patients)
    print patients
    assert n_tr_p + n_de_p + n_te_p < len(patients), \
            "Not enough patients available"
    test_patients = [36]#patients[:n_te_p]
    train_patients = patients[n_te_p:n_te_p+n_tr_p]
    dev_patients = [28]#patients[n_te_p+n_tr_p:n_te_p+n_tr_p+n_de_p]

    if method == 1:
        methods.predict_RF(train_patients, test_patients, fscores,
                           plot_predictions, stratified)
    elif method == 2:
        methods.predict_two_stage(train_patients, test_patients, fscores,
                                  plot_predictions, stratified, dev_pats=dev_patients)
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
        sys.stdout = open(os.path.join('results', "stdout_%s_seed%d_ntrp%d_ntep%d.txt" % (datestr,seed, n_tr_p, n_te_p)), 'w')
    main()
