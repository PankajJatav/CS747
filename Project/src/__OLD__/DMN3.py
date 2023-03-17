import os
import tensorflow as tf
from functools import partial
from preprocess import get_data, vectorize_data, encode_data
import numpy as np
import tensorflow_hub as hub
import itertools

DATAPATH = "./data" #"/home/mike/Desktop/FINAL_PROJECT747/data"
inputpath = 'en/qa2_two-supporting-facts_{}.txt'
inputpath = os.path.join(DATAPATH, inputpath.format("train"))

# x = get_data(inputpath)
module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"

x = get_data(inputpath)

def process_info(info):

    counter = 0
    
    results = ["<START>"]
    end_sent_idx = []

    for sent in info:
        for word in sent:
            results.append(word)
            counter += 1
        results.append("<SEP>")
        counter += 1
        end_sent_idx.append(counter)
        
        
    results.append("<STOP>")
    counter += 1
    end_sent_idx += [counter]*(get_data.max_nsent - len(end_sent_idx))

    # ADD ANOTHER VALUE FOR NUMBER OF SENTENCE -- WILL BE USED AS MEMORY STOP CRITERIA
    end_sent_idx.append(len(info)-1)


    results += ["<PAD>"]*(get_data.max_slen - len(results))

    results =  tuple(map(lambda x: get_data.embedding.get(
        x, get_data.embedding["<UNK>"]), results))


    return tf.concat(
        [tf.convert_to_tensor(results, dtype='int32'), \
         tf.convert_to_tensor(end_sent_idx, dtype='int32')],
        axis=0)


def process_question(question):


    counter = 1

    results = ["<START>"]
    for word in question:
        results.append(word)
        counter += 1
    results.append("<STOP>")
    # counter+= 1

    results += ["<PAD>"]*(get_data.max_qlen - len(results))


    results = tuple(map(lambda x: get_data.embedding.get(
        x, get_data.embedding["<UNK>"]), results))

    # !!! Index of question ending emeded at end, make sure to remove this in training!!!
    results += (counter,)




class DMN(tf.keras.Model):

    def __init__(self, EMBED_SIZE=80, RNN_UNITS=80, MEMORY_SIZE=80, mem_passes=2):
        super().__init__()

        self.embed_layer = tf.keras.layers.Embedding(
            get_data.max_qlen,
            EMBED_SIZE,
            #embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=True,
            input_length=None,
            name="INFO-EMBEDDING-LAYER",
        )

        self.info_gru_layer = tf.keras.layers.GRU(
            RNN_UNITS,
            return_sequences=True,
            return_state=True,
            #stateful=True,
            name="INFO-GRU-LAYER",
            dropout=0.2
        )

        self.question_gru_layer = tf.keras.layers.GRU(
            RNN_UNITS,
            return_sequences=True,
            return_state=True,
            #stateful=True,
            name="QUESTION-GRU-LAYER",
            dropout=0.2
        )

        self.mem_passes=mem_passes
        
        self.mem_attention = tf.keras.layers.Attention(name="MEMORY-ATTENTION", dropout=0.2)
        self.mem_gru = tf.keras.layers.GRU(
            MEMORY_SIZE,
            #len(get_data.vocab),
            return_sequences=True,
            return_state=True,
            #stateful=True,
            name="MEMORY-GRU-LAYER",
            dropout=0.3,
            recurrent_dropout=0.6,
        )        



    def call(self, inputs):

        ## Preprocessor used transfer learning encoder so we don't have to build our own encodings...
        info, question = inputs
        print(len(info), type(info), info.shape)
        print(len(question), type(question), question.shape)

        info_idx = tf.stack([info[i][get_data.max_slen:info.shape[1]-1] for i in range(info.shape[0])])
        num_sent = tf.stack([info[i][info.shape[1]-1:] for i in range(info.shape[0])])

        print(info.shape, info_idx.shape, num_sent.shape)

        padding = tf.tile(num_sent, tf.constant([1, info_idx.shape[1]], tf.int32))
        mask = tf.range(0, limit=info_idx.shape[1])
        zeros = tf.zeros((1, info_idx.shape[1]), tf.int32)
        padding = tf.cast(tf.math.greater_equal((padding - mask), zeros), tf.int32)        
        padding = tf.reshape(padding, (len(info), padding.shape[1], 1))
        padding = tf.cast(padding, tf.float32)


        x = info
        x = self.embed_layer(info)
        hidden, _ = self.info_gru_layer(x)


        y = question


        return hidden



        raise


i = tf.stack(tuple(map(process_info, map(lambda x: x[0], x))))
q = tf.stack(tuple(map(process_question, map(lambda x: x[1], x))))
#process_info(x[0][0])
model = DMN()

qqq = model([i[0:10], i[0:10]])

##lossfn = partial(tf.metrics.categorical_crossentropy, from_logits=True)
##model.compile(optimizer="adam", loss=lossfn)
##model.fit(x=[i, q], y=a, epochs=1, batch_size=10)

