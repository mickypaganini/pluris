import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import json
import tqdm
import keras
import logging

C_FRAC = 0.07
L_FRAC = 0.61

def replaceInfNaN(x, value):
    '''
    replace Inf and NaN with a default value
    Args:
    -----
        x:     arr of values that might be Inf or NaN
        value: default value to replace Inf or Nan with
    Returns:
    --------
        x:     same as input x, but with Inf or Nan raplaced by value
    '''
    x[np.isfinite( x ) == False] = value 
    return x

# -----------------------------------------------------------------  

#def apply_calojet_cuts(df):
def calojet_cuts(df):
    '''
    Apply recommended cuts for Akt4EMTopoJets
    '''
    # are the last two a thing??
    cuts = (abs(df['jet_eta']) < 2.5) & \
           (df['jet_pt'] > 20e3) & \
           (df['jet_aliveAfterOR'] == 1) & \
           ((df['jet_JVT'] > 0.59) | (df['jet_pt'] > 60e3) | (abs(df['jet_eta']) > 2.4)) & \
           (df['jet_aliveAfterORmu'] == 1) & \
           (df['jet_nConst'] > 1) 
    #df = df[cuts].reset_index(drop=True)
    return cuts

# ----------------------------------------------------------------- 

def trk_cuts(df):
    # -- remove jets with 0 tracks
    z0 = df['jet_trk_ip3d_z0']
    notracks = []
    for i in xrange(len(z0)):
        notracks.append(len(z0[i]) == 0)
    cuts = -np.array(notracks)
    return cuts

# ----------------------------------------------------------------- 

def dphi(trk_phi, jet_phi):
    '''
    Definition:
    -----------
        Calculate Delta Phi between track and jet
        The result will be in the range [-pi, +pi]
    Args:
    -----
        trk_phi: series or np.array with the list of track phi's for each jet
        jet_phi: series of np.array with the phi value for each jet
    
    Returns:
    --------
        deltaphi: np.array with the list of values of delta phi between the jet and each track
    '''

    import math
    PI = math.pi
    deltaphi = trk_phi - jet_phi # automatically broadcasts jet_phi across all tracks

    # -- ensure that phi stays within -pi and pi
    for jetN in xrange(len(deltaphi)):
        deltaphi[jetN][deltaphi[jetN] > PI] -= 2*PI
        deltaphi[jetN][deltaphi[jetN] < -PI] += 2*PI

    return deltaphi

# ----------------------------------------------------------------- 

def dr(eta1, eta2, phi1, phi2):
    '''
    Definition:
    -----------
        Function that calculates DR between two objects given their etas and phis
    Args:
    -----
        eta1 = pandas series or array, eta of first object
        eta2 = pandas series or array, eta of second object
        phi1 = pandas series or array, phi of first object
        phi2 = pandas series or array, phi of second object
    Output:
    -------
        dr = float, distance between the two objects 
    '''
    deta = abs(eta1 - eta2)
    dphi = abs(phi1 - phi2)
    dphi = np.array([np.arccos(np.cos(a)) for a in dphi]) # hack to avoid |phi1-phi2| larger than 180 degrees
    return np.array([np.sqrt(pow(de, 2) + pow(dp, 2)) for de, dp in zip(deta, dphi)])

# ----------------------------------------------------------------- 

def athenaname(var):
    '''
    Definition:
    -----------
        Quick utility function to turn variable names into Athena format
        This function may vary according to the names of the variables currently being used
    
    Args:
    -----
        var: a string with the variable name

    Returns:
    --------
        a string with the modified variable name
    '''
    return var.replace('jet_trk_ip3d_', '').replace('jet_trk_', '')

# ----------------------------------------------------------------- 

def sort_tracks(trk, data, theta, grade, sort_by, n_tracks):
    ''' 
    Definition:
        Sort tracks by sort_by and put them into an ndarray called data.
        Pad missing tracks with -999 --> net will have to have Masking layer
    
    Args:
    -----
        trk: a dataframe or pandas serier
        data: an array of shape (nb_samples, nb_tracks, nb_features)
        theta: pandas series with the jet_trk_theta values, used to enforce custom track selection
        grade: numpy array with the grade values
        sort_by: a string representing the column to sort the tracks by
        n_tracks: number of tracks to cut off at. if >, truncate, else, -999 pad
    
    Returns:
    --------
        modifies @a data in place. Pads with -999
    
    '''
    
    for i, jet in tqdm.tqdm(trk.iterrows()):

         # -- remove tracks with grade = -10
        trk_selection = grade[i] != -10 

        # tracks = [[pt's], [eta's], ...] of tracks for each jet 
        tracks = np.array(
                [v[trk_selection].tolist() for v in jet.get_values()],
                dtype='float32'
            )[:, (np.argsort(jet[sort_by][trk_selection]))[::-1]]

        # total number of tracks per jet      
        ntrk = tracks.shape[1] 

        # take all tracks unless there are more than n_tracks 
        data[i, :(min(ntrk, n_tracks)), :] = tracks.T[:(min(ntrk, n_tracks)), :] 

        # default value for missing tracks 
        data[i, (min(ntrk, n_tracks)):, :  ] = -999 

# ----------------------------------------------------------------- 

def scale_tracks(data, var_names, sort_by, file_name, savevars):
    ''' 
    Args:
    -----
        data: a numpy array of shape (nb_samples, nb_tracks, n_variables)
        var_names: list of keys to be used for the model
        sort_by: string with the name of the column used for sorting tracks
                 needed to return json file in format requested by keras2json.py
        file_name: str, tag that identifies the specific way in which the data was prepared, 
                   i.e. '30trk_hits'
        savevars: bool -- True for training, False for testing
                  it decides whether we want to fit on data to find mean and std 
                  or if we want to use those stored in the json file 

    Returns:
    --------
        modifies data in place
    '''    
    logger = logging.getLogger("Track Scaling")
    scale = {}
    if savevars:  
        for v, vname in enumerate(var_names):
            logger.info('Scaling feature %s of %s (%s).' % (v, len(var_names), vname))
            f = data[:, :, v]
            slc = f[f != -999]
            # -- first find the mean and std of the training data
            m, s = slc.mean(), slc.std()
            # -- then scale the training distributions using the found mean and std
            slc -= m
            slc /= s
            data[:, :, v][f != -999] = slc.astype('float32')
            scale[v] = {'name' : athenaname(vname), 'mean' : m, 'sd' : s}

        # -- write variable json for lwtnn keras2json converter
        variable_dict = {
            'inputs' : [],
            'input_sequences' : 
            [{
                'name' : 'track_input',
                'variables' :
                 [{
                    'name': scale[v]['name'],
                    'scale': float(1.0 / scale[v]['sd']),
                    'offset': float(-scale[v]['mean']),
                    'default': None
                    } 
                    for v in xrange(len(var_names))]
            }],
            'outputs': 
            [{
                'name': 'tagging',
                'labels': ['pu', 'pc', 'pb', 'ptau']
            }],
            'keras_version': keras.__version__,
            'miscellaneous': {
                'sort_by': sort_by
            }  
        }
        with open('var' + file_name + '.json', 'wb') as varfile:
            json.dump(variable_dict, varfile)

    # -- when operating on the test sample, use mean and std from training sample
    # -- this info is stored in the json file
    else:
        with open('var' + file_name + '.json', 'rb') as varfile:
            varinfo = json.load(varfile)

        for v, vname in enumerate(var_names):
            logger.info('Scaling feature %s of %s (%s).' % (v, len(var_names), vname))
            f = data[:, :, v]
            slc = f[f != -999]
            ix = [i for i in xrange(len(varinfo['input_sequences'][0]['variables'])) if varinfo['input_sequences'][0]['variables'][i]['name'] == athenaname(vname)]
            offset = varinfo['input_sequences'][0]['variables'][ix[0]]['offset']
            scale = varinfo['input_sequences'][0]['variables'][ix[0]]['scale']
            slc += offset
            slc *= scale
            data[:, :, v][f != -999] = slc.astype('float32')

# ----------------------------------------------------------------- 

def reweight_to_b(X, y, pt_col, eta_col):
    '''
    Definition:
    -----------
        Reweight to b-distribution in eta and pt
    '''
    pt_bins = [10, 50, 100, 150, 200, 300, 500, 99999]
    eta_bins = np.linspace(0, 2.5, 6)

    b_bins = plt.hist2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = plt.hist2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = plt.hist2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wb= np.ones(X[y == 5].shape[0])

    wc = [(b_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], b_bins[2]) - 1
    )]

    wl = [(b_bins[0] / l_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 0, pt_col] / 1000, b_bins[1]) - 1, 
        np.digitize(X[y == 0, eta_col], b_bins[2]) - 1
    )]

    # -- hardcoded, standard flavor fractions
    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = wb 
    w[y == 4] = wc
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 

def reweight_to_l(X, y, pt_col, eta_col):
    '''
    Definition:
    -----------
        Reweight to light-distribution in eta and pt
    '''
    pt_bins = [10, 50, 100, 150, 200, 300, 500, 99999]
    eta_bins = np.linspace(0, 2.5, 6)

    b_bins = plt.hist2d(X[y == 5, pt_col] / 1000, X[y == 5, eta_col], bins=[pt_bins, eta_bins])
    c_bins = plt.hist2d(X[y == 4, pt_col] / 1000, X[y == 4, eta_col], bins=[pt_bins, eta_bins])
    l_bins = plt.hist2d(X[y == 0, pt_col] / 1000, X[y == 0, eta_col], bins=[pt_bins, eta_bins])

    wl= np.ones(X[y == 0].shape[0])

    wc = [(l_bins[0] / c_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 4, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 4, eta_col], l_bins[2]) - 1
    )]

    wb = [(l_bins[0] / b_bins[0])[arg] for arg in zip(
        np.digitize(X[y == 5, pt_col] / 1000, l_bins[1]) - 1, 
        np.digitize(X[y == 5, eta_col], l_bins[2]) - 1
    )]

    # -- hardcoded, standard flavor fractions
    n_light = wl.sum()
    n_charm = (n_light * C_FRAC) / L_FRAC
    n_bottom = (n_light * (1 - L_FRAC - C_FRAC)) / L_FRAC

    w = np.zeros(len(y))
    w[y == 5] = wb 
    w[y == 4] = wc
    w[y == 0] = wl
    return w

# ----------------------------------------------------------------- 