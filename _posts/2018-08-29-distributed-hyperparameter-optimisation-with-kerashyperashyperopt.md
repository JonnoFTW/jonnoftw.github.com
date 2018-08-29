---
layout: post
title: "Distributed Hyperparameter Optimisation with Keras/Theano/Tensorflow/Hyperas/Hyperopt"
description: ""
category: 
tags: [hyperopt, hyperas, keras, mongo]
---
{% include JB/setup %}

# Overview

In this post I will descibe how to use a combination of tools to speed up hyperparameter optimisation tasks. Instructions are provided here for Ubuntu, but could
reasonably be applied to any *nix system.

## What is Hyperparameter Optimisation?

Hyperparameter Optimisation (HO) is a method choosing the optimal paramters for a machine learning task. These parameters include things like:

* Number of layers
* Learning rate
* Batch size
* Layer types
* Dropout
* Optimiser algorithm (SGD, Adam, rmsprop etc.)


It may not be immediately obvious for any given problem what network configuration is best for a given task, so we can use hyperparameter optimisation decide for us by 
intelligently iterating through a search space of the parameters you want to optimise. Hyperopt uses [Tree-Structure Parzen estimators](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) and is quite good at rapidly deciding on an optimal set of parameters. It works by running and evaluating a model, returning a loss score and then running another model with slightly different parameters that aim to minimise the error score.  The hard part for you, is devising a search space for your problem, which may be quite large. To save time, we can run these models simultaneously across any number of machines, and even have each machine run multiple models (provided it has enough cores).
 
 
Luckily there are python libraries that do all this hard work for us!
# Requirements

You will need:

* Python with the following packages installed
  * theano, tensorflow or tensorflow-gpu
  * hyperopt
  * hyperas
  * pymongo 
* Mongodb with a `jobs` database
* A bunch of machines with all of the above installed
* pssh

I strongly recommend using pyenv to use an up to date version of python and to prevent our installed packages from conflicting with system ones.
If you have access to a network drive available to all your machines, set `$PYENV_ROOT` to something they can all see (or at least a common path on all the machines).
Install these using (swap tensorflow with whatever keras backend you want to use, one of: `theano tensorflow-gpu tensorflow cntk`):

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"
curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev
env PYTHON_CONFIGURE_OPTS="--enable-shared" MAKEOPTS="-j 8" pyenv install 3.6.5
pyenv local 3.6.5
pip install tensorflow git+https://github.com/hyperopt/hyperopt git+https://github.com/maxpumperla/hyperas keras pssh matplotlib h5py pymongo
```

You will now have a self contained python install in `$HOME/.pyenv/versions/3.6.5`. Keep in mind that it needs to build a python install and so might take a while. You can copy the `~/.pyenv` folder to any machine you want to run on. Just remember to copy your `~/.bash_profile` (or equivalent) to each machine you want as a worker.


# Code

We also need pick a task to optimise! Hyperas uses templates to generate the code that hyperopt can use, so you need to follow this template closely. Create a file called: `optimise_task.py`. We'll find the optimal layer size 
and dropout parameters (from within the search space, see `l1_size` and `l1_dropout` variables) for a single layer dense network to solve the MNIST task:

```python
from hyperas import optim
from hyperas.distributions import quniform, uniform
from hyperopt import STATUS_OK, tpe, mongoexp
import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tempfile
from datetime import datetime


def data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = 10
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test


def create_model(x_train, y_train, x_test, y_test):
    """
    Create your model...
    """
    l1_size = {{quniform(12, 256, 4)}}
    l1_dropout = {{uniform(0.001, 0.7)}}
    params = {
        'l1_size': l1_size,
        'l1_dropout': l1_dropout
    }
    num_classes = 10
    model = Sequential()
    model.add(Dense(int(l1_size), activation='relu'))
    model.add(Dropout(l1_dropout))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    start = datetime.now()
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    out = {
        'loss': -acc,
        'score': score,
        'status': STATUS_OK,
        'duration': (datetime.now() - start).total_seconds(),
        'ho_params': params,
        'model_config': model.get_config()
    }
    # optionally store a dump of your model here so you can get it from the database later
    temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
    model.save(temp_name)
    with open(temp_name, 'rb') as infile:
        model_bytes = infile.read()
    out['model_serial'] = model_bytes
    return out


if __name__ == "__main__":
    trials = mongoexp.MongoTrials('mongo://username:pass@mongodb.host:27017/jobs/jobs', exp_key='mnist_test')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=trials,
                                          keep_temp=True)  # this last bit is important
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
```

Note the name of the experiment key: `mnist_test`, this will be the key in the `jobs` collection of the `jobs` database in mongodb. After every model has completed, it will be stored in mongodb. It may be 
possible to store the weights in the output document (the output of `model.get_weights()`, but mongodb has a limit of 4MB per document. To get around this, GridFS is used to transprently store blobs in the database of
the model itself.

I've also stored the duration in the result object as well, since you may find 2 models with very similar loss, but the one with slightly better loss may have significantly higher runtime.

# Running

Running this has 2 parts:

1. Trials controller that decides on the parameters that each model run will use
2. Workers that actually run the individual models

## Controller

Run this from a machine (it must be active while all the jobs are running):

```bash
python optimise_task.py
```

You should get an output file called `temp_model.py` (if you don't, make sure you have updated to the latest hyperas code from github). Make sure this file is visible to your workers.

## Worker

These are all your other machines (may also be on the controller machine). Make sure you have pyenv installed on them too, it can be easy to just zip up the .pyenv folder and copy it your home
directory on your worker machines and unzip it. This way you don't miss any dependencies.


Run the following:

```bash
mkdir hyperopt_job
touch hyperopt_job/job.sh
chmod +x hyperopt_job/job.sh
```

Copy the `temp_model.py` file into the hyperopt_job folder  `~/hyperopt_job/job.sh` with:

```bash
#!/bin/bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
export PYTHONPATH=~/hyperopt_job/
cd ~/hyperopt_job
pyenv local 3.6.5
hyperopt-mongo-worker --mongo="mongo://username:password@mongodb.host:27017/jobs" --exp-key=mnist_test
```

Now you can run `~/hyperopt_job/job.sh` on your worker machines! Keep in mind that they need to be able to access the mongodb.
 The workers will continue running jobs until you have reached `max_evals` as defined at the end of `optimise_task.py`.

The problem with this can be that you need to:

* Login to the machine via ssh or regular login
* Start screen
* Start the script

We can do better by using `pssh` to do the above automatically for a list of given hosts.

```bash
pssh -h hosts.txt  bash -c "nohup  ~/hyperopt_job/job.sh &"
```

This will start a start `job.sh` on each host listed in `hosts.txt`. `nohup` will prevent the process from stopping when the ssh session disconnects.

## Results

Once all your jobs are done, you can look through your results using a mongodb browser such as Robo3T. Here's a small script to fetch the model with lowest loss score from the database
and deserialize your model:

```python
from pymongo import MongoClient
from keras.models import load_model
import tempfile
c = MongoClient('mongodb://username:pass@mongodb.host:27017/jobs')
best_model = c['jobs']['jobs'].find_one({'exp_key': 'mnist_test', 'result.status': 'ok'}, sort=[('result.loss', 1)])
temp_name = tempfile.gettempdir()+'/'+next(tempfile._get_candidate_names()) + '.h5'
with open(temp_name, 'wb') as outfile:
    outfile.write(best_model['result']['model_serial'])
model = load_model(temp_name)
# do things with your model here
model.summary()
```


You can use the following to visualise the results with a subset of your search space (typical search spaces with have more than 2 parameters
so it may not be obvious which one (or combination thereof) has the most significant impact on your model performance):

```python
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pymongo import MongoClient
import numpy as np

if __name__ == "__main__":
    # get the data
    jobs = MongoClient('mongodb://username:pass@mongodb.host:27017/jobs')['jobs']['jobs']
    cursor = jobs.find({'exp_key': 'mnist_test', 'result.status': 'ok'}).sort('result.metrics.rmse')
    results = defaultdict(lambda: defaultdict(list))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in ax.get_xticklabels():
        i.set_rotation(90)

    for row in cursor:
        cc = row['tid']
        results[cc]['x'].append(row['result']['loss'])
        results[cc]['y'].append(row['result']['ho_params']['l1_size'])
        results[cc]['z'].append(row['result']['ho_params']['l1_dropout'])

    colors = cm.rainbow(np.linspace(0, 1, len(results)))
    it = iter(colors)
    for k, v in results.items():
        ax.scatter(v['x'], v['y'], v['z'], label=k, color=next(it))
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Layer Count')
    ax.set_zlabel('L1 Size')
    plt.title("Hyperparameter Optimisation Results")
    plt.show()
```


Here's an output from one of my experiments, as you can see, the optimser quickly finds models that cluster around a minimum achievable error score given the search space:

![3d result plot](https://i.imgur.com/lY0Ksaf.png)
