# -*- coding: utf-8 -*-
'''
Info:
    This script can be run directly after
    dataprocess.py. It takes as inputs the 
    HDF5 files produced by the first script 
    and uses them to train a Keras NN for 
    b-tagging.
Author: 
    Michela Paganini - Yale/CERN
    michela.paganini@cern.ch
Example:
    python train.py --input ./data/*.h5 --n_train 193213 --n_test 184022 --n_validate 82811
'''

import pandas as pd
import numpy as np
import math
import os
import sys
import deepdish.io as io
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
import cPickle

def main(hdf5_paths, n_train, n_test, n_validate, model_id):
    '''
    '''

    train_paths = [f for f in hdf5_paths if 'train-' in f]
    test_paths = [f for f in hdf5_paths if 'test-' in f]
    validate_paths = [f for f in hdf5_paths if 'validate-' in f]

    def batch(paths, batch_size, random=True):
        '''
        yield a batch of samples with two inputs (jet and track X),
        three outputs (FC output, RNN output, combined output), and
        three weights corresponding to the three outputs.
        Args:
        -----
            paths: list of paths (strings) to hdf5 files in the format that datatprocessing.py outputs
            batch_size: int, the number of jets in a batch
            random (default=True): bool, whether to shuffle the objects manually
        '''
        while True:
            if random:
                np.random.shuffle(paths)
            for fp in paths:
                d = io.load(fp)
                X = d['X'] # matrix of jet features
                X_trk = d['X_trk'] # matrix of track features
                # transform the label into {0, 1, 2, 3} format
                le = LabelEncoder()
                y = le.fit_transform(d['y'])
                w = d['w'] # array of jet weights
                if random:
                    ix = range(X.shape[0])
                    np.random.shuffle(ix)
                    X, X_trk, y, w = X[ix], X_trk[ix], y[ix], w[ix]
                for i in xrange(int(np.ceil(X.shape[0] / float(batch_size)))):
                    yield (
                        [
                            X[(i * batch_size):((i+1)*batch_size)], 
                            X_trk[(i * batch_size):((i+1)*batch_size)] 
                        ], 
                        [y[(i * batch_size):((i+1)*batch_size)]] * 3,
                        [ 
                            w[(i * batch_size):((i+1)*batch_size)],
                            w[(i * batch_size):((i+1)*batch_size)],
                            w[(i * batch_size):((i+1)*batch_size)]
                        ]
                    )

    def get_n_vars(train_paths):
        '''
        temporarily opens the first hdf5 file just to calculate how many variables
        are used to describe jets and tracks.
        Args:
        -----
            train_paths: list of paths (strings) to hdf5 files in the format that datatprocessing.py outputs
        Returns:
        --------
            number of jet variables
            number of track variables
        '''
        d = io.load(train_paths[0])
        return d['X'].shape[1], d['X_trk'].shape[1:]

    # -- `build_model` is defined below
    net = build_model(*get_n_vars(train_paths))
    net.summary()
 #   plot_model(net,
 #              to_file='model.png',
 #              show_shapes=True,
 #              show_layer_names=False)
    net.compile('adam', 'sparse_categorical_crossentropy')

    weights_path = model_id + '.h5'
    # -- try to find pre-trained net; if not found, start training from beginning
    try:
        print 'Trying to load weights from ' + weights_path
        net.load_weights(weights_path)
        print 'Weights found and loaded from ' + weights_path
    except IOError:
        print 'Could not find weight in ' + weights_path

    # -- train 
    try:
        net.fit_generator(batch(train_paths, 256, random=True),
            steps_per_epoch=np.ceil(n_train/float(256)),
            max_q_size=2,
            workers=2,
            pickle_safe=True,
            verbose=True, 
            #batch_size=64, 
            #sample_weight=train['w'],
            callbacks = [
                EarlyStopping(verbose=True, patience=100, monitor='val_loss'),
                ModelCheckpoint(weights_path, monitor='val_loss', verbose=True, save_best_only=True)
            ],
            epochs=200, 
            validation_data=batch(validate_paths, 2048, random=False),
            validation_steps=np.ceil(n_validate/2048.)
        ) 
    except KeyboardInterrupt:
        print '\n Stopping early.'

    # -- load in best network
    print 'Loading best network...'
    net.load_weights(weights_path)

    # print 'Extracting...'
    # # -- save the predicions
    #np.save('yhat-{}-{}.npy'.format(iptagger, model_id), yhat)

    # from joblib import Parallel, delayed
    # test = Parallel(n_jobs=1, verbose=5, backend="threading")(
    #     delayed(extract)(filepath, ['pt', 'y', 'mv2c10']) for filepath in test_paths
    # )

    # -- test
    print 'Testing...'
    yhat = net.predict_generator(
        batch(test_paths, 2048, random=False),
        steps=int(np.ceil(n_test/2048.)),
        verbose=1)
    np.save('yhat-{}.npy'.format(model_id), yhat) 

    test = [extract(filepath, ['pt', 'y', 'mv2c10']) for filepath in test_paths]

    # -- handle results from parallel predictions (not needed otherwise)
    def dict_reduce(x, y):
        return {
            k: np.concatenate((v, y[k]))
            for k, v in x.iteritems()
        }
    test = reduce(dict_reduce, test)
    test = {k : v[:yhat[-1].shape[0]] for k, v in test.iteritems()}
    
    evaluate_perf(yhat, test, model_id)
    
# -----------------------------------------------------------------

def build_model(n_jet_vars, n_trk_vars):
    '''
    Returns a Keras Model with 2 inputs, 3 outputs
    '''
    from keras.layers import Input, Dense, Dropout, Masking, LSTM, merge
    from keras.layers.advanced_activations import PReLU
    from keras.models import Model

    jet_input = Input(shape=(n_jet_vars,), name='jet_input')

    x1 = Dropout(0.3)(Dense(72, activation='relu')(jet_input))
    x1 = Dropout(0.2)(Dense(72, activation='relu')(x1))
    x1 = Dropout(0.1)(Dense(48, activation='relu')(x1))
    x1 = Dropout(0.1)(Dense(32, activation='relu')(x1))
    y1 = Dense(4, activation='softmax')(x1)

    #jetmodel = Model(inputs=jet_input, outputs=y)

    trk_input = Input(shape=n_trk_vars, name='trk_input')
    x2 = Masking(mask_value=-999)(trk_input)
    #x2 = LSTM(25, return_sequences=True)(x2)
    x2 = LSTM(25, return_sequences=False)(x2)
    x2 = Dropout(0.3)(x2)
    y2 = Dense(4, activation='softmax')(x2)

    m = merge([x1, x2], mode='concat')
    m = Dense(48, activation='softmax')(m)
    m = Dropout(0.2)(m)
    y = Dense(4, activation='softmax')(m)

    return Model(
        inputs=[jet_input, trk_input],
        outputs=[y1, y2, y])

# -----------------------------------------------------------------

def extract(filepath, keys):
    d = io.load(filepath)
    new_d = {k:v for k,v in d.iteritems() if k in keys}
    return new_d

# ----------------------------------------------------------------- 

def build_rocs(yhat, y, mv2c10, model_id, extratitle=''):
    '''
    calculates roc curves, stores them in pickle files,
    plots them and saves them to pdf, and calculates
    bkg rejection at 70% efficiency
    TO-DO: move the 70% efficiency calculation elsewhere
    '''
    from sklearn.metrics import roc_curve
    # -- Select out only pairs of flavors
    bl_sel = (y == 5) | (y == 0)
    cl_sel = (y == 4) | (y == 0) # not used for now
    bc_sel = (y == 5) | (y == 4)

    def _add_curve(name, color, tpr, fpr, dictref):
        '''
        since we want to save out the (x,y) values that make up
        a roc curve, we decide to store them in a dictionary of this type
        '''
        dictref.update(
            {name : {
                    'efficiency' : tpr,
                    'rejection' : 1. / fpr,
                    'color' : color
            }}
        )
    ## bottom versus light
    # -- remove infinities
    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 0])[bl_sel]) 
    # -- init dict that will store b vs light roc curves info
    bl_curves = {} 
    # -- use scikit-learn's roc_curve function to build the roc
    fpr, tpr, _ = roc_curve(
        y[bl_sel][fin1] == 5, # y_true
        np.log(yhat[:, 2] / yhat[:, 0])[bl_sel][fin1]) # y_score
    # -- add the roc to the dictionary called 'bl_curves'
    _add_curve(name='pluris', color='green', tpr=tpr, fpr=fpr, dictref=bl_curves)
    # -- same, for MV2c10 roc curve
    fpr, tpr, _ = roc_curve(
        y[bl_sel] == 5, # y_true
        mv2c10[bl_sel]) # y_score
    _add_curve(name='MV2c10', color='red', tpr=tpr, fpr=fpr, dictref=bl_curves)
    # -- save the roc curves out to a pickle file for future use
    cPickle.dump(bl_curves, open('ROC_' + model_id + '_test_bl.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
    
    def _ROC_plotter(curves, min_eff=0, max_eff=1, linewidth=1.4, pp=False, signal="",
        background="", title="", logscale=True, ymax=10**4, ymin=1):
        '''
        matplotlib stuff to prettify roc curves
        '''
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
        ax = fig.add_subplot(111)
        plt.xlim(min_eff,max_eff)
        plt.grid(b = True, which = 'minor')
        plt.grid(b = True, which = 'major')
        max_ = 0
        for tagger, data in curves.iteritems():
            sel = (data['efficiency'] >= min_eff) & (data['efficiency'] <= max_eff)
            if np.max(data['rejection'][sel]) > max_:
                max_ = np.max(data['rejection'][sel])
            plt.plot(data['efficiency'][sel], data['rejection'][sel], 
                '-', label = r''+tagger, color = data['color'], linewidth=linewidth)
        ax = plt.subplot(1,1,1)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
        plt.ylim(ymin,ymax)
        if logscale == True:
            ax.set_yscale('log')
        ax.set_xlabel(r'$\varepsilon_{\mathrm{signal}}$')
        ax.set_ylabel(r"$1 / \varepsilon_{\mathrm{background}}$")
        plt.legend()
        plt.title(r''+title)
        if pp:
            pp.savefig(fig)
        else:
            return fig

    # -- use the roc curve prettifier above to plot all b vs light curves on the same canvas, and save it
    fg = _ROC_plotter(bl_curves, 
        title=r'pluris vs MV2c10 '+ extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 10000)
    fg.savefig('ROC_' + model_id + '_' + extratitle +'_test_bl.pdf')

    # same, for b versus c
    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 1])[bc_sel])
    bc_curves = {}
    fpr, tpr, _ = roc_curve(
        y[bc_sel][fin1] == 5, # y_true
        np.log(yhat[:, 2] / yhat[:, 1])[bc_sel][fin1]) # y_score
    _add_curve(name='pluris', color='green', tpr=tpr, fpr=fpr, dictref=bc_curves)
    fpr, tpr, _ = roc_curve(
        y[bc_sel] == 5, # y_true
        mv2c10[bc_sel]) # y_score
    _add_curve(name='MV2c10', color='red', tpr=tpr, fpr=fpr, dictref=bc_curves)
    cPickle.dump(bc_curves, open('ROC_' + model_id + '_test_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = _ROC_plotter(bc_curves,
        title=r'pluris vs MV2c10 ' + extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 100)
    fg.savefig('ROC_' + model_id + '_' + extratitle +'_test_bc.pdf')
    plt.close(fg)

    def _find_nearest(array, value):
        return (np.abs(array-value)).argmin()

    return {extratitle : 
        {
            'pluris_70_bl' : bl_curves[r'pluris']['rejection'][_find_nearest(bl_curves[r'pluris']['efficiency'], 0.7)],
            'pluris_70_bc' : bc_curves[r'pluris']['rejection'][_find_nearest(bc_curves[r'pluris']['efficiency'], 0.7)],
            'MV2_70_bl' : bl_curves[r'MV2c10']['rejection'][_find_nearest(bl_curves[r'MV2c10']['efficiency'], 0.7)],
            'MV2_70_bc' : bc_curves[r'MV2c10']['rejection'][_find_nearest(bc_curves[r'MV2c10']['efficiency'], 0.7)]
        }
    }

# ----------------------------------------------------------------- 

def evaluate_perf(yhat, test, model_id):
    '''
    calls the function above to build the roc curves (inclusive and in each pT bin)
    plots profile of performance vs pT at 70% efficiency
    '''
    print 'Plotting...'
    # -- for now, we only care about the final prediction (hence yhat[-1])
    # -- but one could also take a look at how the other outputs are doing
    _ = build_rocs(yhat[-1], test['y'], test['mv2c10'], model_id)

    # -- performance by pT
    print 'Plotting performance in bins of pT...'
    pt_bins = [0, 50000, 100000, 150000, 300000, 500000, max(test['pt'])+1]
    #pt_bins = np.linspace(min(test['pt']), max(test['pt']), 5)
    bn = np.digitize(test['pt'], pt_bins)
    from collections import OrderedDict
    rej_at_70 = OrderedDict()

    for b in np.unique(bn):
        rej_at_70.update(
            build_rocs(
                yhat[-1][bn == b],
                test['y'][bn == b],
                test['mv2c10'][bn == b],
                model_id,
                #'bin{}'.format(b)
                '{}-{}GeV'.format(pt_bins[b-1]/1000, pt_bins[b]/1000)
            )
        )

    # -- find center of each bin:
    bins_mean = [(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)]
    # -- horizontal error bars of lenght = bin length: 
    xerr = [bins_mean[i]-pt_bins[i+1] for i in range(len(bins_mean))]

    # -- build profile of performance vs pT at 70% eff
    def _plot_rej_at70(rej_at_70, bins_mean, xerr, model_id, flavors='bl'):
        '''
        Args:
        -----
            rej_at_70:
            bins_mean:
            xerr:
            model_id:
            flavors: string, one of 'bl', 'bc'

        '''
        plt.clf()
        _ = plt.errorbar(
            bins_mean, 
            [rej_at_70[k]['pluris_70_bl'] for k in rej_at_70.keys()],
            xerr=xerr,
            #yerr=np.sqrt(bin_heights), 
            fmt='o', capsize=0, color='green', label='pluris', alpha=0.7)
        _ = plt.errorbar(
            bins_mean, 
            [rej_at_70[k]['MV2_70_bl'] for k in rej_at_70.keys()],
            xerr=xerr,
            #yerr=np.sqrt(bin_heights), 
            fmt='o', capsize=0, color='red', label='MV2c10', alpha=0.7)
        plt.legend()
        plt.title(r'${}$ rejection at 70% ${}$ efficiency in pT bins'.format(
            flavors[1], flavors[0]))
        plt.yscale('log')
        plt.xlabel(r'$p_{T, \mathrm{jet}} \ \mathrm{MeV}$')
        plt.ylabel('Background rejection at 70% efficiency')
        plt.xlim(xmax=1000000)
        plt.savefig('pt_' + flavors + '_' + model_id + '.pdf')
    
    _plot_rej_at70(rej_at_70, bins_mean, xerr, model_id, flavors='bl')
    _plot_rej_at70(rej_at_70, bins_mean, xerr, model_id, flavors='bc')

# ----------------------------------------------------------------- 

if __name__ == '__main__':
    import argparse
    # -- read in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', required=True, type=int, help="int, Number of training examples")
    parser.add_argument('--n_test', required=True, type=int, help="int, Number of testing examples")
    parser.add_argument('--n_validate', required=True, type=int, help="int, Number of validating examples")
    parser.add_argument('--model_id', default='test', help='str, name used to identify this training')
    parser.add_argument('--input', type=str, nargs="+", help="Path to hdf5 files (e.g.: /path/to/pattern*.h5)")

    args = parser.parse_args()

    sys.exit(main(args.input, args.n_train, args.n_test, args.n_validate, args.model_id))

