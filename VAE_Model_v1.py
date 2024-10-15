import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# Configuration for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
max_length = 120  # Adjust based on your dataset analysis
vocab_size = 50  # Adjust based on your dataset
latent_dim = 50  # Latent space dimension

# Data loading and preprocessing
def load_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles = file.read().splitlines()
    return smiles

def preprocess_smiles(smiles_list):
    tokenizer = Tokenizer(char_level=True, num_words=vocab_size)
    tokenizer.fit_on_texts(smiles_list)
    tensor = tokenizer.texts_to_sequences(smiles_list)
    tensor = pad_sequences(tensor, maxlen=max_length, padding='post')
    return tensor, tokenizer

# Encoder architecture
def build_encoder():
    inputs = layers.Input(shape=(max_length,))
    x = layers.Embedding(input_dim=vocab_size + 1, output_dim=64)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Flatten()(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    model = models.Model(inputs, [z, z_mean, z_log_var])
    return model

# Decoder architecture
def build_decoder():
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(64 * max_length, activation='relu')(latent_inputs)
    x = layers.Reshape((max_length, 64))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    outputs = layers.TimeDistributed(layers.Dense(vocab_size + 1, activation='softmax'))(x)

    model = models.Model(latent_inputs, outputs)
    return model

# Load data
file_path = 'smiles_train.txt'  # Adjust the path as necessary
smiles_list = load_smiles(file_path)
X_train, tokenizer = preprocess_smiles(smiles_list)

# VAE model
encoder = build_encoder()
decoder = build_decoder()
inputs = layers.Input(shape=(max_length,))
z, z_mean, z_log_var = encoder(inputs)
outputs = decoder(z)
vae = models.Model(inputs, outputs)

# VAE loss
reconstruction_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)(inputs, outputs)
reconstruction_loss *= max_length
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1) * -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=Adam(learning_rate=0.001))

# Train the model
vae.fit(X_train, X_train, epochs=1, batch_size=32)

# Optionally, generate new molecules
def generate_smiles(decoder, tokenizer, num_samples=10):
    z_sample = np.random.normal(size=(num_samples, latent_dim))
    pred = decoder.predict(z_sample)
    generated_sequences = np.argmax(pred, axis=-1)
    generated_smiles = tokenizer.sequences_to_texts(generated_sequences)
    return generated_smiles

def save_smiles_to_file(smiles_list, filename='generated_smiles_v1.txt'):
    with open(filename, 'w') as file:
        for smiles in smiles_list:
            file.write(smiles + '\n')

new_smiles = generate_smiles(decoder, tokenizer, num_samples=10000)
save_smiles_to_file(new_smiles)
print("File Saved")
print(new_smiles)
