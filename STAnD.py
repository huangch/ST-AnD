import os
import glob
import random
from random import sample
import numpy as np
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class MILPoolingLayer(layers.Layer):
    def __init__(self, 
                 data_dim=1024, 
                 beta=0.05, 
                 pooling=True, 
                 pooling_type='noisy-and', 
                 pooling_gamma=5.0, 
                 **kwargs):
        super(MILPoolingLayer, self).__init__(**kwargs)
        
        self.data_dim = data_dim
        self.beta = beta
        self.pooling = pooling
        self.pooling_type = pooling_type
        self.pooling_gamma = pooling_gamma
    
        if self.pooling_type == 'noisy-and':
            w_init = tf.random_uniform_initializer()

            self.b = tf.Variable(
                initial_value=w_init(shape=(self.data_dim,), dtype="float32"),
                trainable=True,
                name="NoisyAnd_b",
            )       
            
            
    def get_config(self):
        return super().get_config()
        
        
    def call(self, x):
        # Calculate the mean value across all instances in the bag (batch)
        if self.pooling_type == 'noisy-and':
            mean_pij = tf.math.reduce_mean(x, axis=0) if self.pooling else x            
            noisy_and_A = tf.math.sigmoid(self.pooling_gamma*(mean_pij-self.b))-tf.math.sigmoid(-self.pooling_gamma*self.b)
            noisy_and_B = tf.math.sigmoid(self.pooling_gamma * (1-self.b))-tf.math.sigmoid(-self.pooling_gamma*self.b)
            Pi = noisy_and_A / (noisy_and_B + keras.backend.epsilon())
        
        elif self.pooling_type == 'noisy-or':
            Pi = 1-tf.math.reduce_prod(1-x, axis=0) if self.pooling else x
            
        elif self.pooling_type == 'max':
            Pi = tf.math.reduce_max(x, axis=0) if self.pooling else x
            
        elif self.pooling_type == 'isr':
            isr_A = tf.math.reduce_sum(x/(1-x + keras.backend.epsilon()), axis=0) if self.pooling else x/(1-x + keras.backend.epsilon())
            Pi = isr_A/(1+isr_A+keras.backend.epsilon())
            
        elif self.pooling_type == 'gm':
            gm_A = tf.math.pow(x + keras.backend.epsilon(), self.pooling_gamma)
            gm_B = tf.math.reduce_mean(gm_A, axis=0) if self.pooling else gm_A
            Pi = tf.math.pow(gm_B + keras.backend.epsilon(), 1/self.pooling_gamma)
        
        elif self.pooling_type == 'lse':
            lse_A = tf.math.reduce_mean(tf.math.exp(self.pooling_gamma*x), axis=0) if self.pooling else tf.math.exp(self.pooling_gamma*x)
            Pi = (1/self.pooling_gamma)*tf.math.log(lse_A)

        if self.pooling:
            Pi *= self.beta
            Pi *= tf.cast(tf.shape(x)[0], dtype=tf.float32)                    
            Pi = tf.expand_dims(Pi, axis=0)

        return Pi
    
    
    def set_pooling(self, pooling):        
        self.pooling = pooling
    
    
class DenormalizationLayer(layers.Layer):
    def __init__(self, data_dim=3000, **kwargs):
        super(DenormalizationLayer, self).__init__(**kwargs)
        
        self.data_dim = data_dim
    
        w_init = tf.random_uniform_initializer()

        self.w = tf.Variable(
            initial_value=w_init(shape=(self.data_dim,), dtype="float32"),
            trainable=True,
            name="Denormalization_w",
        )       
            
            
    def get_config(self):
        return super().get_config()
        
        
    def call(self, x):
        # Calculate the mean value across all instances in the bag (batch)
        
        x = tf.multiply(x, self.w)
        x = keras.activations.relu(x)
        outputs = tf.math.log(x+1.0)
         
        return outputs
    
    
class STAnD(keras.models.Model):
    def __init__(self, 
                 input_dim,
                 output_dim,
                 decoder_hidden_layers,                                                
                 fcn_dropout_rate = 0.2,
                 # fcn_batchnormalization = False,
                 enable_coexpr = False,
                 beta = 0.01, 
                 batch_size = 1,
                 pooling = True,
                 pooling_type='noisy-and',
                 pooling_gamma = 5.0,
                 name='stand',
                 **kwargs):
        super(STAnD, self).__init__(**kwargs)
        self.enable_coexpr = enable_coexpr
        self.input_layer = layers.Input(input_dim)
        
        self.cell_fcn_layers = []
        for i, (l) in enumerate(decoder_hidden_layers): 
            self.cell_fcn_layers.append(layers.Dense(l, name="cell_fcn_layer_"+str(i)))
            if fcn_dropout_rate > 0.0: self.cell_fcn_layers.append(layers.Dropout(fcn_dropout_rate, name="cell_fcn_dropout_"+str(i)))
            self.cell_fcn_layers.append(layers.Activation('sigmoid', name="cell_fcn_activation_"+str(i)))
            # if fcn_batchnormalization: self.cell_fcn_layers.append(layers.BatchNormalization(name="cell_fcn_batchnormalization_"+str(i)))

        self.cell_fcn_layers.append(layers.Dense(output_dim, name="cell_fcn_layer_output"))
        if fcn_dropout_rate > 0.0: self.cell_fcn_layers.append(layers.Dropout(fcn_dropout_rate, name="cell_fcn_dropout_output"))
        self.cell_fcn_layers.append(layers.Activation('sigmoid', name="cell_fcn_activation_output"))
        # if fcn_batchnormalization: self.cell_fcn_layers.append(layers.BatchNormalization(name="cell_fcn_batchnormalization_output"))
        
        self.batch_size = batch_size
        self.mil_layer = MILPoolingLayer(data_dim=output_dim, beta=beta, pooling=pooling, pooling_type=pooling_type, pooling_gamma=pooling_gamma, name="mil_layer")
        if self.enable_coexpr: self.coexpr_layer = layers.Dense(output_dim, activation='sigmoid', name="coexpr_layer")
        self.denormalizer = DenormalizationLayer(data_dim = output_dim, name="denormalization_layer")

        self.mse = keras.losses.MeanSquaredError()
        self.loss_tracker = keras.metrics.Mean(name="loss")
        
        self.output_layer = self.call(self.input_layer)
    
        super(STAnD, self).__init__(
            inputs=self.input_layer,
            outputs=self.output_layer,
            **kwargs)
 
    
    def build(self):
        # Initialize the graph
        self._is_graph_network = True
        self._init_graph_network(
            inputs=self.input_layer,
            outputs=self.output_layer
        )
    
    
    def call(self, inputs, training=False):
        x = inputs        
        for fcn_layer in self.cell_fcn_layers: x = fcn_layer(x)
        
        x = self.mil_layer(x)
        if self.enable_coexpr: x = self.coexpr_layer(x)
        outputs = self.denormalizer(x)
        
        return outputs

    
    @property
    def metrics(self):
        return [
            self.loss_tracker,
        ]

    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss = tf.constant(0.0, dtype=tf.float32)
            for b in range(self.batch_size):
                reduced_sum = tf.math.reduce_sum(tf.abs(x[b]), axis=1)
                zero = tf.zeros(shape=(1,1), dtype=tf.float32)
                nonzero_mask = tf.not_equal(reduced_sum, zero)
                z = tf.boolean_mask(x[b], nonzero_mask[0])
                z = self.call(z)
                total_loss += self.mse(z, y[b])
            
            loss = total_loss / self.batch_size
            
        # Backpropagation.
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Loss tracking.
        self.loss_tracker.update_state(loss)

        # Log results.
        return {
            "loss": self.loss_tracker.result(),
        }
       
        
    def test_step(self, data):
        x, y = data
        total_loss = tf.constant(0.0, dtype=tf.float32)
        for b in range(self.batch_size):
            reduced_sum = tf.math.reduce_sum(tf.abs(x[b]), axis=1)
            zero = tf.zeros(shape=(1,1), dtype=tf.float32)
            nonzero_mask = tf.not_equal(reduced_sum, zero)
            z = tf.boolean_mask(x[b], nonzero_mask[0])
            z = self.call(z)
            total_loss += self.mse(z, y[b])            
        
        loss = total_loss / self.batch_size

        # Loss tracking.
        self.loss_tracker.update_state(loss)

        # Log results.
        return {
            "loss": self.loss_tracker.result(),
            }          

    
    def set_pooling(self, pooling):
        self.mil_layer.set_pooling(pooling)
        
        
class STAnDDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 image_folder, 
                 sptx_data, 
                 gene_list, 
                 vqvae_encoder, 
                 vqvae_quantizer, 
                 batch_size = 1,
                 augmentation = True,
                 random_seed=1234, 
                 k=5, 
                 m_list=[0,1,2,3,4]):
        'Initialization'
        
        self.image_folder = image_folder
        self.vqvae_encoder = vqvae_encoder
        self.vqvae_quantizer = vqvae_quantizer
        self.batch_size = batch_size
        self.augmentation = augmentation
        
        loc_list = [os.path.basename(name) for name in glob.glob(os.path.join(self.image_folder, '*'))]
        loc_list.sort()
        random.seed(random_seed)            
        random.shuffle(loc_list) 
        loc_list_partitions = self.__partition(loc_list, k)
        self.loc_list = [i for m in m_list for i in loc_list_partitions[m] if i in sptx_data.obs_names.tolist()]   
        self.sptx_data = sptx_data[self.loc_list, gene_list]                
        self.on_epoch_end()
        
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.loc_list) / self.batch_size))
        
        
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.loc_list[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        random.shuffle(self.loc_list)        

        
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X_list = []
        y_list = []

        mx_len = 0
        
        for loc in indexes:
            l = len(glob.glob(os.path.join(self.image_folder, loc, '*.png')))           
            mx_len = l if l > mx_len else mx_len
            
        for loc in indexes:
            image_list = glob.glob(os.path.join(self.image_folder, loc, '*.png'))
            
            enc_input_list = []
            for fn in image_list:
                img = Image.open(fn)
                ary = np.array(img)
                
                if self.augmentation:
                    if random.randint(0, 1) == 1:
                        ary = np.flipud(ary)
                    if random.randint(0, 1) == 1:
                        ary = np.fliplr(ary)            
                    if random.randint(0, 1) == 1:
                        ary = np.transpose(ary, axes=(1,0,2))    
            
                ary = np.expand_dims(ary, axis=0)
                enc_input_list.append(ary)
    
            enc_inputs = np.concatenate(enc_input_list, axis=0)
            enc_inputs = enc_inputs.astype(np.float32) / 255.0 - 0.5        
            
            enc_outputs = self.vqvae_encoder.predict(enc_inputs)
            flat_enc_outputs = enc_outputs.reshape(-1, enc_outputs.shape[-1])
            latent = self.vqvae_quantizer.get_quantized_latent_value(flat_enc_outputs)
            latent = latent.numpy().reshape((enc_outputs.shape[0], enc_outputs.shape[1], enc_outputs.shape[2], enc_outputs.shape[3]))
            
            data = latent.reshape(enc_outputs.shape[0], enc_outputs.shape[1]*enc_outputs.shape[2]*enc_outputs.shape[3])  
            padding = np.zeros(shape=(mx_len - data.shape[0], data.shape[1]))
            X = np.concatenate((data, padding))
            
            y = self.sptx_data[[loc], :].to_df().to_numpy()
            
            X_list.append(X)
            y_list.append(y)
        
        return np.stack(X_list), np.stack(y_list)

    
    def __partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]