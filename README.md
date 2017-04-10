# pluris

A pluri-stream, deep-learning solution for ATLAS flavor tagging.

At the moment, the model is designed to have two streams: a recurrent stream which processes track information, and a fully-connected stream that processes the outputs of taggers such as `JetFitter`, `SV1` and `SMT`.
We can obtain 3 separate outputs from the neural net: the output of each stream and the output of the merged representations.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

This project was built using `Keras v2.0.2` with the TensorFlow backend (`TF v1.0.1`). Install Keras using `pip`, and augment it with the TensorFlow backend by following the instructions [here](https://www.tensorflow.org/install/).

Another major requirement for the data processing step is [`ROOT`](https://root.cern.ch/), with `pip`-installable `rootpy` and `root_numpy`.
The `.root` files will be transformed into `HDF5` format, so you will need to `pip install h5py deepdish`. 

Other requirements can be satified with `pip install tqdm pandas numpy scikit-learn matplotlib` and the `viz` package found [here](https://github.com/mickypaganini/IPRNN/tree/master/viz).

## How To Run

1. Get access to simulated ttbar samples identified by the tag `group.perf-flavtag` which contain track level information.
2. Construct the feature matrices `X` and `X_trk`, target array `y`, and weights array `w` using [`dataprocessing.py`](dataprocessing.py).
This will produce HDF5 files containing a dictionary for training, validating and final testing.
The remaining scripts will use the `.h5` output produced here. 
The variables that will be inserted in the final dataset are specified in [`variables.yaml`](variables.yaml).
There is quite a lot going on in this step:
    * jets are selected using [`calojet_cuts`](https://github.com/mickypaganini/pluris/blob/master/data_utils.py#L30) and [`trk_cuts`](https://github.com/mickypaganini/pluris/blob/master/data_utils.py#L46)
    * jets are reweighted in pT and eta using [`reweight_to_l`](https://github.com/mickypaganini/pluris/blob/master/data_utils.py#L288)
    * tracks are sorted in decreasing order of the `--sort_by` argument (see usage below),
    * cut to include a maximum number of tracks per jet specified by the `--ntrk` argument,
    * and padded with value -999 for shorter sequences
    * finally jets are scaled using the `scikit-learn` `StandardScaler` and randomly assigned to thr train, test and validate datasets using `train_test_split`. 
```
usage: dataprocessing.py [-h] [--variables VARIABLES] [--model_id MODEL_ID]
                         [--sort_by SORT_BY] [--ntrk NTRK]
                         [--input INPUT [INPUT ...]]

optional arguments:
  -h, --help            show this help message and exit
  --variables VARIABLES
                        path to the yaml file containing lists named
                        'branches', 'training_vars', 'trk_vars'
  --model_id MODEL_ID   token to identify the model
  --sort_by SORT_BY     str, name of the variable used to order tracks in an
                        event
  --ntrk NTRK           Maximum number of tracks per event. If the event has
                        fewer tracks, use padding; if is has more, only
                        consider the first ntrk
  --input INPUT [INPUT ...]
                        Path to root files, e.g. /path/to/pattern*.root
```
3. Run the training and evaluate the performance using [`train.py`](train.py).
The model architecture is defined in [`build_model`](https://github.com/mickypaganini/pluris/blob/master/train.py#L216).
As it's designed now, this script also includes the plotting functions to build ROC curves and evaluate performance in pT bins at 70% efficiency.
These should probably be refactored.
```
usage: train.py [-h] --n_train N_TRAIN --n_test N_TEST --n_validate N_VALIDATE
                [--model_id MODEL_ID] [--input INPUT [INPUT ...]]

optional arguments:
  -h, --help            show this help message and exit
  --n_train N_TRAIN     int, Number of training examples
  --n_test N_TEST       int, Number of testing examples
  --n_validate N_VALIDATE
                        int, Number of validating examples
  --model_id MODEL_ID   str, name used to identify this training
  --input INPUT [INPUT ...]
                        Path to hdf5 files (e.g.: /path/to/pattern*.h5)
```

## Authors

* **Michela Paganini** - *Initial work* - [mickypaganini](https://github.com/mickypaganini)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
