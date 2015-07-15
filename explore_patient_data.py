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

def run_experiment(method):
    seed = 98234111

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
    #patients = np.random.permutation(193) + 1
    n_tr_p = 50 # Train patients
    n_de_p = 0 # Development patients
    n_te_p = 100 # Test patients
    assert n_tr_p + n_de_p + n_te_p < len(patients), \
            "Not enough patients available"
    #train_patients = patients[:n_tr_p]
    #test_patients = patients[n_tr_p:n_tr_p+n_te_p]
    #dev_patients = patients[n_tr_p+n_te_p:n_tr_p+n_te_p+n_de_p]
    test_patients = patients[:n_te_p]
    train_patients = patients[n_te_p:n_te_p+n_tr_p]
    dev_patients = patients[n_te_p+n_tr_p:n_te_p+n_tr_p+n_de_p]

    plot_predictions = True
    stratified = False
    if method == 1:
        methods.predict_RF(train_patients, test_patients, fscores,
                           plot_predictions, stratified)
    elif method == 2:
        methods.predict_two_stage(train_patients, test_patients, fscores,
                                  plot_predictions, stratified)
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
    sys.stdout = open(os.path.join('results', "stdout_%s_seed%d.txt" % (datestr,seed)), 'w')
    main()
