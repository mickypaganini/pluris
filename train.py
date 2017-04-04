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
    python train1.py --input ./data/*.h5 --n_train 193213 --n_test 184022 --n_validate 82811
'''

import pandas as pd
import pandautils as pup
import numpy as np
import math
import os
import sys
import deepdish.io as io
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from viz import add_curve, calculate_roc, ROC_plotter
import cPickle

def main(hdf5_paths, n_train, n_test, n_validate, model_id):
    '''
    '''

    train_paths = [f for f in hdf5_paths if 'train-' in f]
    test_paths = [f for f in hdf5_paths if 'test-' in f]
    validate_paths = [f for f in hdf5_paths if 'validate-' in f]

    def batch(paths, batch_size, random=True):
        while True:
            if random:
                np.random.shuffle(paths)
            for fp in paths:
                d = io.load(fp)
                X = d['X']
                X_trk = d['X_trk']
                le = LabelEncoder()
                y = le.fit_transform(d['y'])
                w = d['w']
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
                            0.4 * w[(i * batch_size):((i+1)*batch_size)],
                            0.4 * w[(i * batch_size):((i+1)*batch_size)],
                            w[(i * batch_size):((i+1)*batch_size)]
                        ]
                    )

    def get_n_vars(train_paths):
        d = io.load(train_paths[0])
        return d['X'].shape[1], d['X_trk'].shape[1:]

    net = build_model(*get_n_vars(train_paths))
    net.summary()
    plot_model(net,
               to_file='model.png',
               show_shapes=True,
               show_layer_names=False)
    net.compile('adam', 'sparse_categorical_crossentropy')

    weights_path = model_id + '.h5'
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

    print 'Extracting...'
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

    test = [extract(filepath, ['pt', 'y', 'mv2c10']) for filepath in test_paths]

    def dict_reduce(x, y):
        return {
            k: np.concatenate((v, y[k]))
            for k, v in x.iteritems()
        }

    test = reduce(dict_reduce, test)
    test = {k : v[:yhat[-1].shape[0]] for k, v in test.iteritems()}

    print 'Plotting...'
    # -- for now, we only care about the final prediction
    _ = performance(yhat[-1], test['y'], test['mv2c10'], model_id)

    # -- Performance by pT
    # print 'Plotting performance in bins of pT...'
    # #pt_bins = [0, 50000, 100000, 150000, 200000, 300000, 500000, max(test['pt'])+1]
    # pt_bins = np.linspace(min(test['pt']), max(test['pt']), 5)
    # bn = np.digitize(test['pt'], pt_bins)
    # from collections import OrderedDict
    # rej_at_70 = OrderedDict()

    # for b in np.unique(bn):
    #     rej_at_70.update(
    #         performance(
    #             yhat[-1][bn == b],
    #             test['y'][bn == b],
    #             test['mv2c10'][bn == b],
    #             '{}-{}GeV'.format(pt_bins[b-1]/1000, pt_bins[b]/1000)
    #         )
    #     )

    # # -- find center of each bin:
    # bins_mean = [(pt_bins[i]+pt_bins[i+1])/2 for i in range(len(pt_bins)-1)]
    # # -- horizontal error bars of lenght = bin length: 
    # xerr = [bins_mean[i]-pt_bins[i+1] for i in range(len(bins_mean))]

    # plt.clf()
    # _ = plt.errorbar(
    #     bins_mean, 
    #     [rej_at_70[k]['MST_70_bl'] for k in rej_at_70.keys()],
    #     xerr=xerr,
    #     #yerr=np.sqrt(bin_heights), 
    #     fmt='o', capsize=0, color='green', label='MST', alpha=0.7)
    # _ = plt.errorbar(
    #     bins_mean, 
    #     [rej_at_70[k]['MV2_70_bl'] for k in rej_at_70.keys()],
    #     xerr=xerr,
    #     #yerr=np.sqrt(bin_heights), 
    #     fmt='o', capsize=0, color='red', label='MV2c10', alpha=0.7)
    # plt.legend()
    # plt.title('b vs. l rejection at 70% efficiency in pT bins')
    # plt.yscale('log')
    # plt.xlabel(r'$p_{T, \mathrm{jet}} \ \mathrm{MeV}$')
    # plt.ylabel('Background rejection at 70% efficiency')
    # plt.xlim(xmax=1000000)
    # plt.savefig('pt_bl.pdf')

    # plt.clf()
    # _ = plt.errorbar(
    #     bins_mean, 
    #     [rej_at_70[k]['MST_70_bc'] for k in rej_at_70.keys()],
    #     xerr=xerr,
    #     #yerr=np.sqrt(bin_heights), 
    #     fmt='o', capsize=0, color='green', label='MST', alpha=0.7)
    # _ = plt.errorbar(
    #     bins_mean,
    #     [rej_at_70[k]['MV2_70_bc'] for k in rej_at_70.keys()],
    #     xerr=xerr,
    #     #yerr=np.sqrt(bin_heights), 
    #     fmt='o', capsize=0, color='red', label='MV2c10', alpha=0.7)
    # plt.legend()
    # plt.title('b vs. c rejection at 70% efficiency in pT bins')
    # plt.xlabel(r'$p_{T, \mathrm{jet}} \ \mathrm{MeV}$')
    # plt.ylabel('Background rejection at 70% efficiency')
    # plt.yscale('log')
    # plt.xlim(xmax=1000000)
    # plt.savefig('pt_bc.pdf')

    
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

def performance(yhat, y, mv2c10, model_id, extratitle=''):
    # -- Find flavors after applying cuts:
    bl_sel = (y == 5) | (y == 0)
    cl_sel = (y == 4) | (y == 0)
    bc_sel = (y == 5) | (y == 4)

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 0])[bl_sel])
    bl_curves = {}
    add_curve(r'MST', 'green', 
          calculate_roc( y[bl_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 0])[bl_sel][fin1]),
          bl_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( y[bl_sel] == 5, mv2c10[bl_sel]),
          bl_curves)
    cPickle.dump(bl_curves, open('ROC_' + model_id + '_test_bl.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bl_curves, title=r'MST vs MV2c10 '+ extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 10000)
    fg.savefig('ROC_' + model_id + '_' + extratitle +'_test_bl.pdf')

    fin1 = np.isfinite(np.log(yhat[:, 2] / yhat[:, 1])[bc_sel])
    bc_curves = {}
    add_curve(r'MST', 'green', 
          calculate_roc( y[bc_sel][fin1] == 5, np.log(yhat[:, 2] / yhat[:, 1])[bc_sel][fin1]),
          bc_curves)
    add_curve(r'MV2c10', 'red', 
          calculate_roc( y[bc_sel] == 5, mv2c10[bc_sel]),
          bc_curves)
    cPickle.dump(bc_curves, open('ROC_' + model_id + '_test_bc.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)

    fg = ROC_plotter(bc_curves, title=r'MST vs MV2c10 ' + extratitle, min_eff = 0.5, max_eff=1.0, logscale=True, ymax = 100)
    fg.savefig('ROC_' + model_id + '_' + extratitle +'_test_bc.pdf')
    plt.close(fg)

    def find_nearest(array, value):
        return (np.abs(array-value)).argmin()

    return {extratitle : 
        {
            'MST_70_bl' : bl_curves[r'MST']['rejection'][find_nearest(bl_curves[r'MST']['efficiency'], 0.7)],
            'MST_70_bc' : bc_curves[r'MST']['rejection'][find_nearest(bc_curves[r'MST']['efficiency'], 0.7)],
            'MV2_70_bl' : bl_curves[r'MV2c10']['rejection'][find_nearest(bl_curves[r'MV2c10']['efficiency'], 0.7)],
            'MV2_70_bc' : bc_curves[r'MV2c10']['rejection'][find_nearest(bc_curves[r'MV2c10']['efficiency'], 0.7)]
        }
    }

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

