'''
Info:
    This script loads root files containing standard 
    b-tagging information + track level info. 
    It replicates the variable creation and modification
    that are present in MV2, scales the variables 
    and splits them into training and testing sets.
    Finally, the data is stored as dictionaries in
    HDF5 format. Parallelized using joblib.
Author: 
    Michela Paganini - Yale
    michela.paganini@cern.ch
Example:
     
'''
import glob
import pandas as pd
import numpy as np
import math
import os
import sys
import tqdm
import logging
import yaml
import deepdish.io as io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# -- custom utility functions defined in this folder
from utils import configure_logging
from data_utils import replaceInfNaN, calojet_cuts, trk_cuts, reweight_to_l, scale_tracks, sort_tracks

def main(yaml_file, root_paths, model_id, sort_by, n_tracks):
    '''
    Args:
    -----
        
    Returns:
    --------
        n_train_events: total number of events for training, across all root files
        n_test_events: total number of events for testing, across all root files
        n_validate_events: total number of events for validating, across all root files
    Alternatively, you can return the paths to hdf5 files being created, for logging
    '''

    # -- logging
    configure_logging()
    logger = logging.getLogger("// data generation")
    logger.debug('Files to process: {}'.format(root_paths))

    # -- open and process files in parallel
    from joblib import Parallel, delayed
    n_events = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing") \
        (delayed(process)(i, filepath, yaml_file, model_id, sort_by, n_tracks) 
            for i, filepath in enumerate(root_paths))

    # -- add up events in the list of results to get the total number of events per type 
    n_train_events = sum(zip(*n_events)[0])
    n_test_events = sum(zip(*n_events)[1])
    n_validate_events = sum(zip(*n_events)[2])
    logger.info('--n_train {n_train_events} --n_test {n_test_events} --n_validate {n_validate_events}'.format(
            n_train_events=n_train_events,
            n_test_events=n_test_events,
            n_validate_events=n_validate_events
            )
        )
    return n_train_events, n_test_events, n_validate_events

    # -- Alternatively, you can return the paths to hdf5 files being created, for logging
    # hdf5_paths = Parallel(n_jobs=-1, verbose=5, backend="multiprocessing") \
    #     (delayed(f)(i, filepath) for i, filepath in enumerate(root_paths))
    # logger.debug('Saved the following hdf5 archives: {}'.format(hdf5_paths))
    # return hdf5_paths

# -----------------------------------------------------------------

def process(i, filepath, yaml_file, model_id, sort_by, n_tracks): 
    '''
    '''   
    import pandautils as pup

    # -- load branches from yaml file
    branches, training_vars, trk_vars = set_features(yaml_file)
    logger = logging.getLogger("ETL Service")

    # -- load root file to dataframe
    logger.info('Operating on {}'.format(filepath))
    logger.info('Creating dataframes...')
    df = pup.root2panda(filepath, 'bTag_AntiKt4EMTopoJets', branches=branches)
    trk_df = pup.root2panda(filepath, 'bTag_AntiKt4EMTopoJets', branches=trk_vars)

    # -- create MV2 input quantities, set default values
    logger.info('Transforming variables...')
    df = transformVars(df)

    # -- flatten to jet-flat structure
    logger.info('Flattening df...')
    df.drop(['PVx', 'PVy', 'PVz'], axis=1, inplace=True)
    df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in df.iteritems()})
    trk_df_flat = pd.DataFrame({k: pup.flatten(c) for k, c in trk_df.iteritems()})
    del df, trk_df

    # -- jet selection
    logger.info('Applying cuts...')
    cut1 = calojet_cuts(df_flat)
    cut2 = trk_cuts(trk_df_flat)
    df_flat = df_flat[(cut1 & cut2) == 1].reset_index(drop=True)
    trk_df_flat = trk_df_flat[(cut1 & cut2) == 1].reset_index(drop=True)

    # -- create numpy arrays for ML
    logger.info('Creating X, y, w, mv2c10...')
    y = df_flat['jet_LabDr_HadF'].values
    mv2c10 = df_flat['jet_mv2c10'].values
    jet_pt = df_flat['jet_pt'].values

    # -- slice df by only keeping the training variables
    X = df_flat[training_vars].values

    # -- Find weights by reweighting to the light distribution
    pteta = df_flat[['jet_pt', 'abs(jet_eta)']].values
    w = reweight_to_l(pteta, y, pt_col=0, eta_col=1)
    del df_flat, pteta

    # -- separate theta from the other variables because we don't train on it. Ugly hack :((
    theta = trk_df_flat['jet_trk_theta']
    grade = trk_df_flat['jet_trk_ip3d_grade'].values
     # -- drop variables from the df that we do not want to use for training
    trk_df_flat.drop([
        'jet_trk_ip3d_grade'
        ],
        axis=1, inplace=True) # no longer needed - not an input 
    n_variables = trk_df_flat.shape[1] 
    var_names = trk_df_flat.keys().tolist()
    trk_X = np.zeros((trk_df_flat.shape[0], n_tracks, n_variables), dtype='float32')
    # -- call functions to build X (= trk_X) and sorted_grade                                                                                                                                                                     
    sort_tracks(trk_df_flat, trk_X, theta, grade, sort_by, n_tracks)

    # -- shuffle data, split into train and test
    logger.info('Shuffling, splitting, scaling...')
    ix = np.array(range(len(y)))
    X_train, X_test,\
    trk_X_train, trk_X_test,\
    y_train, y_test,\
    w_train, w_test,\
    ix_train, ix_test, \
    mv2c10_train, mv2c10_test,\
    jet_pt_train, jet_pt_test = train_test_split(
        X, trk_X, y, w, ix, mv2c10, jet_pt, train_size=0.6
    )

    # -- scale inputs to 0 mean, 1 std
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    logger.info('Scaling features ...')
    scale_tracks(trk_X_train, var_names, sort_by, model_id, savevars=True)
    scale_tracks(trk_X_test, var_names, sort_by, model_id, savevars=False)
    # -- default values                                                                                                                                                                                          
    trk_X_train[np.isnan(trk_X_train)] = -999
    trk_X_test[np.isnan(trk_X_test)] = -999

    # -- split the previously selected training data into train and validate
    X_train, X_validate,\
    trk_X_train, trk_X_validate,\
    y_train, y_validate,\
    w_train, w_validate,\
    ix_train, ix_validate,\
    mv2c10_train, mv2c10_validate,\
    jet_pt_train, jet_pt_validate = train_test_split(
        X_train, trk_X_train, y_train, w_train, ix_train, mv2c10_train, jet_pt_train, train_size=0.7
    )

    # -- assign train, test, validate data to dictionaries
    train = {
        'X' : X_train,
        'X_trk' : trk_X_train,
        'y' : y_train,
        'w' : w_train,
        'ix': ix_train,
        'mv2c10': mv2c10_train,
        'pt': jet_pt_train
    }

    test = {
        'X' : X_test,
        'X_trk' : trk_X_test,
        'y' : y_test,
        'w' : w_test,
        'ix': ix_test,
        'mv2c10': mv2c10_test,
        'pt': jet_pt_test
    }

    validate = {
        'X' : X_validate,
        'X_trk' : trk_X_validate,
        'y' : y_validate,
        'w' : w_validate,
        'ix': ix_validate,
        'mv2c10': mv2c10_validate,
        'pt': jet_pt_validate
    }

    # -- save dictionaries to hdf5
    logger.info('Saving dictionaries to hdf5...')
    hdf5_train_path = os.path.join('.', 'data', 'data-' + model_id + str(i) +'-train-db.h5')
    hdf5_test_path = os.path.join('.', 'data', 'data-' + model_id + str(i) +'-test-db.h5')
    hdf5_validate_path = os.path.join('.', 'data', 'data-' + model_id + str(i) +'-validate-db.h5')

    io.save(hdf5_train_path, train)
    io.save(hdf5_test_path, test)
    io.save(hdf5_validate_path, validate)
    logger.debug('Saved hdf5 archives: {}, {}, {}'. format(hdf5_train_path, hdf5_test_path, hdf5_validate_path))

    return (y_train.shape[0], y_test.shape[0], y_validate.shape[0])
    #return (hdf5_train_path, hdf5_test_path, hdf5_validate_path)

# -----------------------------------------------------------------    

def set_features(yaml_file):
    '''
    Info:
    -----
        Load names of branches to use from a yaml file
        This will contain 4 entries: 'branches', 'training_vars', 'ip3d_training_vars', 'ipmp_training_vars'
        - 'branches': list of names of the branches to directly extract from the TTree
        - 'training_vars': list of names of variables to always be used for learning
        - 'ip3d_training_vars': list of names of variables to be used for\
        learning only if we want to include the ip3d vars
        - 'ipmp_training_vars': list of names of variables to be used for\
        learning only if we want to include the ipmp vars
    Returns:
    --------
    '''
    with open(yaml_file, 'r') as stream:
        try:
            s = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return s['branches'], s['training_vars'], s['trk_vars']

# ----------------------------------------------------------------- 

def transformVars(df):
    '''
    modifies the variables to create the ones that mv2 uses, inserts default values when needed, saves new variables
    in the dataframe
    Args:
    -----
        df: pandas dataframe containing all the interesting variables as extracted from the .root file
    Returns:
    --------
        modified mv2-compliant dataframe
    '''
    from rootpy.vector import LorentzVector, Vector3
    import pandautils as pup

    # -- modify features and set default values
    df['abs(jet_eta)'] = abs(df['jet_eta'])

    # -- SV1 features
    dx = df['jet_sv1_vtx_x']-df['PVx']
    dy = df['jet_sv1_vtx_y']-df['PVy']
    dz = df['jet_sv1_vtx_z']-df['PVz']

    v_jet = LorentzVector()
    pv2sv = Vector3()
    sv1_L3d = []
    sv1_Lxy = []
    dR = [] 

    for index, dxi in enumerate(dx): # loop thru events
        sv1_L3d_ev = []
        sv1L_ev = []
        dR_ev = []
        for jet in xrange(len(dxi)): # loop thru jets
            v_jet.SetPtEtaPhiM(df['jet_pt'][index][jet], df['jet_eta'][index][jet], df['jet_phi'][index][jet], df['jet_m'][index][jet])
            if (dxi[jet].size != 0):
                sv1_L3d_ev.append(np.sqrt(pow(dx[index][jet], 2) + pow(dy[index][jet], 2) + pow(dz[index][jet], 2))[0])
                sv1L_ev.append(math.hypot(dx[index][jet], dy[index][jet]))
                
                pv2sv.SetXYZ(dx[index][jet], dy[index][jet], dz[index][jet])
                jetAxis = Vector3(v_jet.Px(), v_jet.Py(), v_jet.Pz())
                dR_ev.append(pv2sv.DeltaR(jetAxis))
            else: 
                dR_ev.append(-1)   
                sv1L_ev.append(-100)
                sv1_L3d_ev.append(-100)
             
        sv1_Lxy.append(sv1L_ev)
        dR.append(dR_ev) 
        sv1_L3d.append(sv1_L3d_ev)
        
    df['jet_sv1_dR'] = dR 
    df['jet_sv1_Lxy'] = sv1_Lxy
    df['jet_sv1_L3d'] = sv1_L3d

    # -- add more default values for sv1 variables
    sv1_vtx_ok = pup.match_shape(np.asarray([len(el) for event in df['jet_sv1_vtx_x'] for el in event]), df['jet_pt'])

    for (ok4event, sv1_ntkv4event, sv1_n2t4event, sv1_mass4event, sv1_efrc4event, sv1_sig34event) in zip(sv1_vtx_ok, df['jet_sv1_ntrkv'], df['jet_sv1_n2t'], df['jet_sv1_m'], df['jet_sv1_efc'], df['jet_sv1_sig3d']): 
        sv1_ntkv4event[np.asarray(ok4event) == 0] = -1
        sv1_n2t4event[np.asarray(ok4event) == 0] = -1 
        sv1_mass4event[np.asarray(ok4event) == 0] = -1000
        sv1_efrc4event[np.asarray(ok4event) == 0] = -1 
        sv1_sig34event[np.asarray(ok4event) == 0] = -100

    # -- JF features
    jf_dR = []
    for eventN, (etas, phis, masses) in enumerate(zip(df['jet_jf_deta'], df['jet_jf_dphi'], df['jet_jf_m'])): # loop thru events
        jf_dR_ev = []
        for m in xrange(len(masses)): # loop thru jets
            if (masses[m] > 0):
                jf_dR_ev.append(np.sqrt(etas[m] * etas[m] + phis[m] * phis[m]))
            else:
                jf_dR_ev.append(-10)
        jf_dR.append(jf_dR_ev)
    df['jet_jf_dR'] = jf_dR

    # -- add more default values for jf variables
    for (jf_mass,jf_n2tv,jf_ntrkv,jf_nvtx,jf_nvtx1t,jf_efrc,jf_sig3) in zip(df['jet_jf_m'],df['jet_jf_n2t'],df['jet_jf_ntrkAtVx'],df['jet_jf_nvtx'],df['jet_jf_nvtx1t'],df['jet_jf_efc'],df['jet_jf_sig3d']):
        jf_n2tv[jf_mass <= 0] = -1;
        jf_ntrkv[jf_mass <= 0] = -1;
        jf_nvtx[jf_mass <= 0]  = -1;
        jf_nvtx1t[jf_mass <= 0]= -1;
        jf_mass[jf_mass <= 0]  = -1e3;
        jf_efrc[jf_mass <= 0]  = -1;
        jf_sig3[jf_mass <= 0]  = -100;

    return df

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    # Look at the yaml file in this directory for an example
    parser.add_argument('--variables',
        type=str, help="path to the yaml file containing lists named\
        'branches', 'training_vars', 'rnnip_training_vars'") # my design choice
    parser.add_argument('--model_id',
        help="token to identify the model",
        default='')
    parser.add_argument('--sort_by', 
        default='jet_trk_ip3d_d0', 
        help='str, name of the variable used to order tracks in an event')
    parser.add_argument('--ntrk', 
        default=20, type=int, 
        help="Maximum number of tracks per event. \
        If the event has fewer tracks, use padding; if is has more, only consider the first ntrk")
    parser.add_argument('--input', 
        type=str, nargs="+", help="Path to root files, e.g. /path/to/pattern*.root")
    args = parser.parse_args()

    sys.exit(main(args.variables, args.input, args.model_id, args.sort_by, args.ntrk))


