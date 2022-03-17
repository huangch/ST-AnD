import os
import glob
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = (
            beta  # This parameter is best kept between [0.25, 2] as per the paper.
        )

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )

        
    def call(self, x):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = self.beta * tf.reduce_mean(
            (tf.stop_gradient(quantized) - x) ** 2
        )
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    
    def get_code_indices(self, flattened_inputs):
        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices
    
    
    def get_quantized_latent_value(self, flattened_inputs):
        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(flattened_inputs)
        flattened = tf.reshape(flattened_inputs, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        return quantized    

    
def get_encoder(image_size=28, latent_dim=16):
    encoder_inputs = keras.Input(shape=(image_size, image_size, 3))
    x = encoder_inputs
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(latent_dim, 1, padding="same")(x)
    encoder_outputs = x
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16):
    latent_inputs = keras.Input(shape=get_encoder().output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def get_vqvae(image_size=28, latent_dim=16, num_embeddings=1024):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(image_size, latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(image_size, image_size, 3))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class VQVAETrainer(keras.models.Model):
    def __init__(self, train_variance, image_size=28, latent_dim=16, num_embeddings=1024, **kwargs):
        super(VQVAETrainer, self).__init__(**kwargs)
        self.train_variance = train_variance
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings

        self.vqvae = get_vqvae(self.image_size, self.latent_dim, self.num_embeddings)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.vq_loss_tracker = keras.metrics.Mean(name="vq_loss")

   
    def call(self, inputs):
        return self.vqvae.call(inputs)

    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.vq_loss_tracker,
        ]

    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            # Outputs from the VQ-VAE.
            reconstructions = self.vqvae(x)

            # Calculate the losses.
            reconstruction_loss = (
                tf.reduce_mean((x - reconstructions) ** 2) / self.train_variance
            )
            total_loss = reconstruction_loss + sum(self.vqvae.losses)

        # Backpropagation.
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))

        # Loss tracking.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.vq_loss_tracker.update_state(sum(self.vqvae.losses))

        # Log results.
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "vqvae_loss": self.vq_loss_tracker.result(),
        }
        
        
class VQVAEDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 image_folder,                   
                 batch_size,
                 augmentation = True,
                 random_seed=1234,
                 k=5, 
                 m_list=[0,1,2,3,4]):
        'Initialization'
        
        self.batch_size = batch_size
        self.image_folder = image_folder
        self.augmentation = augmentation
        
        # loc_list = []
        # for name in glob.glob(os.path.join(self.image_folder, '*')):
        #     loc = os.path.basename(name)
        #     loc_list.append(loc)
        loc_list = [os.path.basename(name) for name in glob.glob(os.path.join(self.image_folder, '*'))]
    
        loc_list.sort()
        random.seed(random_seed)            
        random.shuffle(loc_list) 

        loc_list_partitions = self.__partition(loc_list, k)
        
        # selected_loc_list = []        
        # for m in m_list: selected_loc_list += loc_list_partitions[m]
        selected_loc_list = [i for m in m_list for i in loc_list_partitions[m]]
            
        # self.image_list = []
        # for loc in selected_loc_list:
        #     self.image_list += glob.glob(os.path.join(image_folder, loc, '*.png'), recursive=True)
        self.image_list = [i for loc in selected_loc_list for i in glob.glob(os.path.join(image_folder, loc, '*.png'), recursive=True) ]
        
        self.on_epoch_end()
    
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.image_list) / self.batch_size))

    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        image_list = self.image_list[index*self.batch_size:(index+1)*self.batch_size]
        
        # Generate data
        X = self.__data_generation(image_list)
        
        return X

    
    def on_epoch_end(self):
        'Updates indexes after each epoch'        
        random.shuffle(self.image_list)        

        
    def __data_generation(self, image_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
                
        X_list = []
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
            X_list.append(ary)

        X = np.concatenate(X_list, axis=0)
        X = X.astype(np.float32) / 255.0 - 0.5
        
        return X

    
    def __partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n) ]

    
    def get_variance(self):
        image_list = self.image_list if len(self.image_list) < 10000 else random.choices(self.image_list, k=10000)
        
        X = self.__data_generation(image_list)
        
        var = np.var(X+0.5)
        return var