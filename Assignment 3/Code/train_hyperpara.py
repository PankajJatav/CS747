#!/usr/bin/env python
# coding: utf-8

# ___
#
# ___
# # Text Generation with Neural Networks
#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.__version__

# ## Step 1: The Data
#
# You can grab any free text you want from here: https://www.gutenberg.org/
#
# We'll choose all of shakespeare's works (which we have already downloaded for you), mainly for two reasons:
#
# 1. Its a large corpus of text, its usually recommended you have at least a source of 1 million characters total to get realistic text generation.
#
# 2. It has a very distinctive style. Since the text data uses old style english and is formatted in the style of a stage play, it will be very obvious to us if the model is able to reproduce similar results.


path_to_file = 'shakespeare.txt'

text = open(path_to_file, 'r').read()

print(text[:500])

# ### Understanding unique characters


# The unique characters in the file
vocab = sorted(set(text))
print(vocab)
len(vocab)

# ## Step 2: Text Processing

# ### Text Vectorization
#
# We know a neural network can't take in the raw string data, we need to assign numbers to each character. Let's create two dictionaries that can go from numeric index to character and character to numeric index.


char_to_ind = {u: i for i, u in enumerate(vocab)}

print(char_to_ind)

ind_to_char = np.array(vocab)

print(ind_to_char)

encoded_text = np.array([char_to_ind[c] for c in text])

print(encoded_text)

# We now have a mapping we can use to go back and forth from characters to numerics.


sample = text[:20]

print(sample)

print(encoded_text[:20])

# ## Step 3: Creating Batches
#
# Overall what we are trying to achieve is to have the model predict the next highest probability character given a historical sequence of characters. Its up to us (the user) to choose how long that historic sequence is. Too short a sequence and we don't have enough information (e.g. given the letter "a" , what is the next character?) , too long a sequence and training will take too long and most likely overfit to sequence characters that are irrelevant to characters farther out. While there is no correct sequence length choice, you should consider the text itself, how long normal phrases are in it, and a reasonable idea of what characters/words are relevant to each other.


print(text[:500])

line = "From fairest creatures we desire increase"

len(line)

part_stanza = """From fairest creatures we desire increase,
  That thereby beauty's rose might never die,
  But as the riper should by time decease,"""

len(part_stanza)

# ### Training Sequences
#
# The actual text data will be the text sequence shifted one character forward. For example:
#
# Sequence In: "Hello my nam"
# Sequence Out: "ello my name"
#
#
# We can use the `tf.data.Dataset.from_tensor_slices` function to convert a text vector into a stream of character indices.


seq_len = 120

total_num_seq = len(text) // (seq_len + 1)

total_num_seq

# Create Training Sequences
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for i in char_dataset.take(500):
    print(ind_to_char[i.numpy()])

# The **batch** method converts these individual character calls into sequences we can feed in as a batch. We use seq_len+1 because of zero indexing. Here is what drop_remainder means:
#
# drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
#     whether the last batch should be dropped in the case it has fewer than
#     `batch_size` elements; the default behavior is not to drop the smaller
#     batch.
#


sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)


# Now that we have our sequences, we will perform the following steps for each one to create our target text sequences:
#
# 1. Grab the input text sequence
# 2. Assign the target text sequence as the input text sequence shifted by one step forward
# 3. Group them together as a tuple


def create_seq_targets(seq):
    input_txt = seq[:-1]
    target_txt = seq[1:]
    return input_txt, target_txt


dataset = sequences.map(create_seq_targets)

for input_txt, target_txt in dataset.take(1):
    print(input_txt.numpy())
    print(''.join(ind_to_char[input_txt.numpy()]))
    print('\n')
    print(target_txt.numpy())
    # There is an extra whitespace!
    print(''.join(ind_to_char[target_txt.numpy()]))

# ### Generating training batches
#
# Now that we have the actual sequences, we will create the batches, we want to shuffle these sequences into a random order, so the model doesn't overfit to any section of the text, but can instead generate characters given any seed text.


# Batch size
batch_size = 128

# Buffer size to shuffle the dataset so it doesn't attempt to shuffle
# the entire sequence in memory. Instead, it maintains a buffer in which it shuffles elements
buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

dataset

# ## Step 4: Creating the Model

# This is where YOU will create the model. it needs to start with an embedding layer.
#
# The embedding layer will serve as the input layer, which essentially creates a lookup table that maps the numbers indices of each character to a vector with "embedding dim" number of dimensions. As you can imagine, the larger this embedding size, the more complex the training. This is similar to the idea behind word2vec, where words are mapped to some n-dimensional space. Embedding before feeding straight into the LSTM or GRU usually leads to more realisitic results.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, GRU

# ### Setting up Loss Function
#
# For our loss we will use sparse categorical crossentropy, which we can import from Keras. We will also set this as logits=True


from tensorflow.keras.losses import sparse_categorical_crossentropy


# The reason we need to redefine this is to make sure we are using one hot encoding (from_logits=True)
def sparse_cat_loss(y_true, y_pred):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

# dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def create_model(vocab_size, embed_dim, rnn_neurons, batch_size):
    # define your model here...
    # don't forget you need an embeddings layer at the beginning
    # and a dense layer the size of the vocabulary at the end to generate distributions

    model = Sequential()
    model.add(Embedding(vocab_size, embed_dim, batch_input_shape=[batch_size, None]))
    model.add(LSTM(rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
    # Final Dense Layer to Predict
    model.add(Dense(vocab_size))
    model.compile(optimizer='adam', loss=sparse_cat_loss, metrics=['accuracy'])
    return model


def generate_text(model, start_seed, gen_size=100, temp=1.0):
    '''
    model: Trained Model to Generate Text
    start_seed: Intial Seed text in string form
    gen_size: Number of characters to generate

    Basic idea behind this function is to take in some seed text, format it so
    that it is in the correct shape for our network, then loop the sequence as
    we keep adding our own predicted characters. Similar to our work in the RNN
    time series problems.
    '''

    # Number of characters to generate
    num_generate = gen_size

    # Vecotrizing starting seed text
    input_eval = [char_to_ind[s] for s in start_seed]

    # Expand to match batch format shape
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty list to hold resulting generated text
    text_generated = []

    # Temperature effects randomness in our resulting text
    # The term is derived from entropy/thermodynamics.
    # The temperature is used to effect probability of next characters.
    # Higher probability == lesss surprising/ more expected
    # Lower temperature == more surprising / less expected

    temperature = temp

    # Here batch size == 1
    model.reset_states()

    for i in range(num_generate):
        # Generate Predictions
        predictions = model(input_eval)

        # Remove the batch shape dimension
        predictions = tf.squeeze(predictions, 0)

        # Use a cateogircal disitribution to select the next character
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted charracter for the next input
        input_eval = tf.expand_dims([predicted_id], 0)

        # Transform back to character letter
        text_generated.append(ind_to_char[predicted_id])

    return (start_seed + ''.join(text_generated))


for neu in [64, 128, 256, 512, 1024]:

    # Length of the vocabulary in chars
    vocab_size = len(vocab)

    # The embedding dimension. Your choice.
    # The embedding dimension. Your choice.
    embed_dim = 64

    # Number of RNN units. Your choice. YOU MUST EXPERIMENT WITH THIS NUMBER.
    rnn_neurons = neu

    # Now let's create a function that easily adapts to different variables as shown above.

    # create your model here, passing the parameters
    model = create_model(
      vocab_size = vocab_size,
      embed_dim=embed_dim,
      rnn_neurons=rnn_neurons,
      batch_size=batch_size)


    #Train the model
    epochs = 30
    model.fit(dataset,epochs=epochs)

    losses = pd.DataFrame(model.history.history)
    print(losses.head())

    losses[['loss', 'accuracy']].plot()

    ax = losses.plot(lw=2, colormap='jet', marker='.', markersize=10, title='Loss and Accuracy trend during training '+str(neu) )
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.figure.savefig('L_'+str(neu)+'.png')
    model.save('shakespeare_gen_L_'+str(neu)+'.h5')

    for input_example_batch, target_example_batch in dataset.take(1):
        # Predict off some random batch
        example_batch_predictions = model(input_example_batch)

        # Display the dimensions of the predictions
        print(example_batch_predictions.shape, " <=== (batch_size, sequence_length, vocab_size)")
    print(example_batch_predictions)

    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    print(sampled_indices)

    # Reformat to not be a lists of lists
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()

    print(sampled_indices)

    print("Given the input seq: \n")
    print("".join(ind_to_char[input_example_batch[0]]))
    print('\n')
    print("Next Char Predictions: \n")
    print("".join(ind_to_char[sampled_indices]))
    model = create_model(vocab_size, embed_dim, rnn_neurons, batch_size=1)
    model.load_weights('shakespeare_gen.h5')
    model.build(tf.TensorShape([1, None]))
    model.summary()
    print(generate_text(model,"JULIET",gen_size=1000))
    print(generate_text(model,"BUT",gen_size=1000))
