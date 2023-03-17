import os
import tensorflow as tf
from functools import partial
from preprocess import get_data # ,vectorize_data, encode_data
import numpy as np


DATAPATH = "./data" #"/home/mike/Desktop/FINAL_PROJECT747/data"
inputpath = 'en/qa2_two-supporting-facts_{}.txt'
inputpath = os.path.join(DATAPATH, inputpath.format("train"))

x = get_data(inputpath)



# One-hot encode answers based on vocab
def process_ans(answer):

    idx = get_data.embedding.get(answer, get_data.embedding["<UNK>"])

    encoding = [0.0]*len(get_data.vocab)
    encoding[idx] = 1.0

    return tf.convert_to_tensor(encoding, dtype='float32')


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


    return tf.convert_to_tensor(results, dtype='int32')

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



class DMN(tf.keras.Model):

    def __init__(self, EMBED_SIZE=50, RNN_UNITS=32, MEMORY_SIZE=64, mem_passes=3):
        super().__init__()

        self.info_embed_layer = tf.keras.layers.Embedding(
            get_data.max_slen,
            EMBED_SIZE,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            name="INFO-EMBEDDING-LAYER"
        )

        self.info_gru_layer = tf.keras.layers.GRU(
            RNN_UNITS,
            return_sequences=True,
            return_state=True,
            stateful=True,
            name="INFO-GRU-LAYER",
            dropout=0.2
        )

        self.question_embed_layer = tf.keras.layers.Embedding(
            get_data.max_qlen,
            EMBED_SIZE,
            embeddings_initializer='uniform',
            embeddings_regularizer=None,
            activity_regularizer=None,
            embeddings_constraint=None,
            mask_zero=False,
            input_length=None,
            name="QUESTION-EMBEDDING-LAYER",
        )


        ## We only want to take the last output,
        ##   but this is not the actaul last output due to padding... with have to parse using gather_nd
        self.question_gru_layer = tf.keras.layers.GRU(
            RNN_UNITS,
            return_sequences=True,
            return_state=True,
            stateful=True,
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
            stateful=True,
            name="MEMORY-GRU-LAYER",
            dropout=0.3,
            recurrent_dropout=0.6,
        )

        self.ans_softmax = tf.keras.layers.Softmax(name="ANSWER-SOFTMAX")
        #self.ans_softmax = tf.keras.layers.Activation('sigmoid', name="ANSWER-SIGMOID")
        self.ans_gru = tf.keras.layers.GRU(
            len(get_data.vocab),
            return_sequences=True,
            return_state=True,
            stateful=True,
            name="ANSWER-GRU",
            dropout=0.3,
            recurrent_dropout=0.6,
            #activation="softmax"
        )


    def call(self, inputs):

        ## First decompose the input -- Model can only take a single input tensor...
        info_index = get_data.max_slen + get_data.max_nsent + 1
        question_index = get_data.max_qlen + 1


        input_info = tf.stack([inputs[i][:info_index] for i in range(inputs.shape[0])])
        input_questions = tf.stack([inputs[i][info_index:] for i in range(inputs.shape[0])])

        ### INPUT MODULE ###

        info = tf.stack([input_info[i][:get_data.max_slen] for i in range(input_info.shape[0])])
        info_idx = tf.stack([input_info[i][get_data.max_slen:input_info.shape[1]-1] for i in range(input_info.shape[0])])
        num_sent = tf.stack([input_info[i][input_info.shape[1]-1:] for i in range(input_info.shape[0])])

        x = self.info_embed_layer(info)
        hidden, _ = self.info_gru_layer(x)

        # Only take hidden layers that are at end of sentences
        results = []

        for b in range(len(input_info)):
            #nd = list(map(lambda x: [x], list(info_idx[b].numpy())))
            nd = tf.reshape(info_idx[b], (info_idx.shape[1], 1))
            results.append(tf.gather_nd(hidden[b], nd))
        encoded_info = tf.stack(results)
        


        ### QUESTION MODULE ###
        question = tf.stack([input_questions[i][:get_data.max_qlen] for i in range(len(input_questions))])
        question_idx = tf.stack([input_questions[i][get_data.max_qlen:] for i in range(len(input_questions))])

        x = self.question_embed_layer(question)
        hidden, _ = self.question_gru_layer(x)

        # Only take hidden layers that are at end of question (SHOULD ONLY BE 1)
        results = []
        for b in range(len(input_questions)):
            #nd = list(map(lambda x: [x], list(question_idx[b].numpy())))
            nd = tf.reshape(question_idx[b], (question_idx.shape[1], 1))
            results.append(tf.gather_nd(hidden[b], nd))

        encoded_question = tf.stack(results)

        return encoded_info

        ### MEMORY MODULE ###
        for _ in range(self.mem_passes):
            #attention = self.mem_attention([encoded_info, encoded_question])
            attention = encoded_info
            hidden, _ = self.mem_gru(attention)


        results = []
        for b in range(len(num_sent)):
            nd = tf.reshape(tf.constant([1], dtype='int32'), (num_sent.shape[1], 1))
            #nd = tf.reshape(num_sent[b]-1, (num_sent.shape[1], 1))
            results.append(tf.gather_nd(hidden[b], nd))
        c = tf.stack(results)

        return c



        ### ANSWER MODULE ###
        t_encoded_question = tf.tile(encoded_question, tf.constant([1, hidden.shape[1], 1], tf.int32))
        c = tf.concat([hidden, t_encoded_question], axis=2)
        mem_out = hidden

##
##

        return c
        

        #c = self.ans_softmax(c)
        #hidden, out = self.ans_gru(c)
        hidden, out = self.ans_gru(c)

        results = []
        for b in range(len(num_sent)):
            # nd = tf.reshape(tf.constant([1], dtype='int32'), (num_sent.shape[1], 1))
            nd = tf.reshape(num_sent[b]-1, (num_sent.shape[1], 1))
            results.append(tf.gather_nd(hidden[b], nd))
        c = tf.stack(results)


        c = self.ans_softmax(c)
        return c



        print(hidden.shape, out.shape)

        #out = self.ans_softmax(out)

        out = tf.reshape(out, (out.shape[0], 1, out.shape[1]))
        return out

        return tf.stack(results)

        
        

        

##        # We want to only take the hidden layer at the index of the number of input sentences (num_sent)
##        results = []
##        for b in range(len(num_sent)):
##            #nd = list(map(lambda x: [x], list(num_sent[b].numpy())))
##            nd = tf.reshape(num_sent[b], (num_sent.shape[1], 1))
##            results.append(tf.gather_nd(hidden[b], nd))
##        mem_out = tf.stack(results)
##
##        print(mem_out.shape)
##
##        ### ANSWER MODULE ###
##
##        # Concat last memory output (hidden memory layer at index of number of sentences) and encoded question
##        c = tf.concat([mem_out, encoded_question], axis=2)
##        #c = mem_out
##
##        # x = self.ans_softmax(c)
##        hidden, out = self.ans_gru(x)
##
##        return hidden
##        
##        qqq = []
##        for b in range(2):#len(hidden)):
##            print(hidden[b].shape)
##
##
##            for x in range(len(hidden[b])):
##                print(hidden[b][x])
##
##            print("NEXT OUT")
##
##        raise


        


i = tf.stack(tuple(map(process_info, map(lambda x: x[0], x))))
q = tf.stack(tuple(map(process_question, map(lambda x: x[1], x))))
a = tf.stack(tuple(map(process_ans, map(lambda x: x[2], x))))
a = tf.reshape(a, [len(a), 1, len(get_data.vocab)])


#i = tf.stack(tuple(map(process_info, map(lambda x: x[0], x))))

##inputs = i
##info = tf.stack([inputs[i][:get_data.max_slen] for i in range(len(inputs))])
##idx = tf.stack([inputs[i][get_data.max_slen:] for i in range(len(inputs))])
##imodel = InfoEncoder()
##a = imodel(i[0:10])
##qmodel = QuestionEncoder()
##b = qmodel(q[0:10])
##
##mem = MemoryUnit()
##mi = tf.concat([a, b], 1)
##
##mo = mem(mi)
##
###bb = tf.stack([b for _ in range(46)], axis=1)
##
##t = tf.constant([1,46, 1], tf.int32)
##bb = tf.tile(b, t)
##c = tf.concat([mo,bb], 2)
##
##
##ans = AnswerUnit()
##out = ans(c)

total_inputs = tf.concat([i, q], 1)
lossfn = partial(tf.metrics.categorical_crossentropy, from_logits=True)

d = DMN()


optim = tf.keras.optimizers.SGD(
    learning_rate=0.001,
    momentum=0.005,
    nesterov=True,
    name='SGD',
)

d.compile(optimizer='adam', loss=lossfn)

#d.fit(x=total_inputs, y=a, batch_size=10, epochs=10, shuffle=False)
dout = d(total_inputs[0:10])

