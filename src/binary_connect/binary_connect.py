# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import utils

import os
LONESTAR_MACHINE = 'LONESTAR_MACHINE' in os.environ


def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)


# The binarization function
def binarization(W,
                 H,
                 binary=True,
                 deterministic=False,
                 stochastic=False,
                 srng=None,
                 ternary=False,
                 zero_threshold=0.15):

    if binary and ternary:
        raise Exception('Cannot be binary and ternary at the same time!')

    # (deterministic == True) <-> test-time <-> inference-time
    if (not (binary or ternary)) or (deterministic and stochastic):
        # print("not binary")
        Wb = W
    elif binary:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        
        if stochastic:
            # Stochastic BinaryConnect
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)),
                        theano.config.floatX)
        else:
            # Deterministic BinaryConnect (round to nearest)
            Wb = T.round(Wb)
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    else:
        # Ternary
        if stochastic:
            raise Exception('TERNARY weights with stochastic training'
                            'not supported yet!')
        else:
            Wb = T.switch(T.lt(T.abs_(W), zero_threshold),
                          0.,
                          T.switch(T.lt(W, 0.), -1., 1.))
            Wb = T.cast(Wb, theano.config.floatX)
    
    return Wb


# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    def __init__(self,
                 incoming,
                 num_units,
                 binary = True,
                 stochastic = True,
                 H=1.,
                 W_LR_scale="Glorot",
                 ternary=False,
                 zero_threshold=0.15,
                 **kwargs):
        
        self.binary = binary
        self.ternary = ternary
        self.zero_threshold = zero_threshold
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary or self.ternary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)


    def get_output_for(self, input, deterministic=False, **kwargs):
        self.Wb = binarization(self.W,
                               self.H,
                               self.binary,
                               deterministic,
                               self.stochastic,
                               self._srng,
                               self.ternary,
                               self.zero_threshold)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue


# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self,
                 incoming,
                 num_filters,
                 filter_size,
                 binary = True,
                 stochastic = True,
                 H=1.,
                 W_LR_scale="Glorot",
                 **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    def convolve(self, input, deterministic=False, **kwargs):

        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss, network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            # print("W_LR_scale = "+str(layer.W_LR_scale))
            # print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates


# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default train function in Lasagne yet)
def train(train_fn,
          val_fn,
          batch_size,
          LR_start,LR_decay,
          num_epochs,
          X_train,y_train,
          X_val,y_val,
          X_test,y_test,
          network,
          return_best_epoch,
          callback_after_epoch=None):
    
    # A function which shuffles a dataset
    def shuffle(X,y):
    
        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)
        # print(shuffled_range[0:10])
        
        new_X = np.copy(X)
        new_y = np.copy(y)
        
        for i in range(len(X)):
            
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]
            
        return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X,y,LR):
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            if not LONESTAR_MACHINE:
                loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size], LR)
            else:
                # There was some error with downcasting on Lonestar machines;
                # thus explicitly casting LR to float32.
                loss += train_fn(X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size], np.float32(LR))
        
        loss/=batches
        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X,y):
        
        err = 0
        loss = 0
        batches = len(X)/batch_size
        
        for i in range(batches):
            new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            err += new_err
            loss += new_loss
        
        err = err / batches * 100
        loss /= batches

        return err, loss

    # TODO: Create copy of X_train, y_train initially
    # and do not create copy in every shuffle operation.

    # shuffle the train set
    X_train,y_train = shuffle(X_train,y_train)
    test_err = None
    best_val_err = 100
    best_epoch = 1
    LR = LR_start

    training_losses = []
    validation_losses = []
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        
        start_time = time.time()

        train_loss = train_epoch(X_train,y_train,LR)
        training_losses.append(train_loss)
        X_train,y_train = shuffle(X_train,y_train)
        val_err, val_loss = val_epoch(X_val,y_val)
        validation_losses.append(val_loss)
        if callback_after_epoch:
            callback_after_epoch(epoch)
        
        # test if validation error went down
        if val_err <= best_val_err:
            best_val_err = val_err
            best_epoch = epoch+1
            
            test_err, test_loss = val_epoch(X_test,y_test)
            if return_best_epoch:
                params = lasagne.layers.get_all_param_values(network)

        epoch_duration = time.time() - start_time

        if epoch % 1 == 0:
            # Then we print the results for this epoch:
            print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
            print("  LR:                            {0:.6f}".format(LR))
            print("  training loss:                 {0:.6f}".format(train_loss))
            print("  validation loss:               {0:.6f}".format(val_loss))
            print("  validation error rate:         {0:.6f}".format(val_err)+"%")
            print("  best epoch:                    {0}".format(best_epoch))
            print("  best validation error rate:    {0:.6f}".format(best_val_err)+"%")
            if test_err:
                print("  best epoch test loss:          {0:.6f}".format(test_loss))
                print("  best epoch test error rate:    {0:.6f}".format(test_err)+"%")
        
        # decay the LR
        LR *= LR_decay

    results = {}
    if return_best_epoch:
        lasagne.layers.set_all_param_values(network, params)
        results['best_epoch'] = best_epoch

    results['training_losses'] = training_losses
    results['validation_losses'] = validation_losses
    results['test_error'] = test_err

    weight_counts = utils.count_weights(network)

    results['#total'] = weight_counts['#total']
    results['#zero'] = weight_counts['#zero']
    results['#one'] = weight_counts['#one']
    results['#minusone'] = weight_counts['#minusone']

    return results
