import os
import string
import itertools
import pickle

import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub


DATAPATH = "./data" #"/home/mike/Desktop/FINAL_PROJECT747/data"
inputpath = 'en/qa2_two-supporting-facts_{}.txt'
inputpath = os.path.join(DATAPATH, inputpath.format("train"))


def get_data(path):

    #get_data.vocab = None
    get_data.max_slen = None
    get_data.max_qlen = None
    get_data.max_nsent = None
    get_data.end_sent_idx = None

    def tokenize(line):
        # tokenize.vocab.add("<STOP>")
        def _():
            for part in line.split():
                if(part[-1] in string.punctuation):
                    yield part[:-1]
                    yield part[-1]
                else:
                    yield part
        _ = list(_())
        for __ in _:
            if(__.isnumeric()):
                continue
            tokenize.vocab.add(__)
        return _       
    tokenize.vocab = set()

    def parse_answer(line):

        tokes = tokenize(line.strip())[1:]
        
        q_idx = tokes.index("?")+1

        question = tokes[:q_idx]
        answer = tokes[q_idx]

        return question, answer


    with open(path, 'r') as r:

        current_idx, max_slen, max_qlen, max_nsent = 0, 0, 0, 0
        history, results = [], []

        end_sent_idx = {}
        
        for line in r.readlines():
            line_idx = int(line.split()[0])

            if(line_idx < current_idx or "?" in line):              

                question, answer = parse_answer(line)
                results.append([history, question, answer])

                max_slen = max([max_slen, sum(map(len, history))])#max([max_slen, len(history)])#sum(map(len, history[:-1]))])
                max_qlen = max([max_qlen, sum(map(len, question))])

                max_nsent = max([max_nsent, len(history)])

                for i, word in enumerate(itertools.chain(*history)):
                    if(word == "."):
                        try:
                            end_sent_idx[len(results)-1].append(i)
                        except KeyError:
                            end_sent_idx[len(results)-1] = []
                            end_sent_idx[len(results)-1].append(i)

                current_idx = 0
                history = []

            else:
                tokes = tokenize(line.strip())[1:]              
                history.append(tokes)
                current_idx += 1

    get_data.vocab = tokenize.vocab
    get_data.max_slen = max_slen
    get_data.max_qlen = max_qlen
    get_data.max_nsent= max_nsent
    get_data.embedding = dict(zip(get_data.vocab, range(1, 1+len(get_data.vocab))))

    get_data.embedding["<STOP>"] = 0
    get_data.embedding["<START>"] = -1
    get_data.embedding["<UNK>"] = -2
    get_data.embedding["<SEP>"] = -3
    get_data.embedding["<PAD>"] = -4

    get_data.vocab.add("<STOP>")
    get_data.vocab.add("<START>")
    get_data.vocab.add("<UNK>")
    get_data.vocab.add("<SEP>")
    get_data.vocab.add("<PAD>")
    
    get_data.end_sent_idx = end_sent_idx

    # Account for added START, STOP, and SEP tokens
    get_data.max_slen = get_data.max_slen + get_data.max_nsent + 2
    get_data.max_qlen += 2
    
    return results
#get_data.vocab = None
get_data.max_slen = None
get_data.max_qlen = None
#get_data.embedding = None

def vectorize_data(inputpath):

    inputs = get_data(inputpath)
    vectorize_data.inputs = inputs

    max_num_sentences = max(map(lambda x: len(x[0]), inputs))

    info = np.zeros((len(inputs), get_data.max_slen))
    questions = np.zeros((len(inputs), get_data.max_qlen))


    ans = np.zeros((len(inputs), 1+len(get_data.vocab)))
    embedding = dict(zip(get_data.vocab, range(1, 1+len(get_data.vocab))))
    vectorize_data.embedding = embedding

    for i, row in enumerate(inputs):
        sentences, question, answer = row
        for j, word in enumerate(itertools.chain(*sentences)):
            info[i][j] = embedding[word]
        for j, word in enumerate(question):
            questions[i][j] = embedding[word]

        #print(ans.shape)

        try:
            ans[i][embedding[answer]] = 1.0
        except IndexError:
            print(i, answer, embedding[answer], embedding, len(embedding))
            raise
        
        #print(ans[i])
        #raise
        #ans[i][0] = embedding[answer]
    
    return tf.convert_to_tensor(info), \
           tf.convert_to_tensor(questions), \
           tf.convert_to_tensor(ans)
vectorize_data.inputs = None
vectorize_data.embedding = None

#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"

def encode_data(inputpath, overwrite=False):

    try:
        if(overwrite):
            raise FileNotFoundError("OVERWRITE {}.pickle".format(inputpath))

        _ = vectorize_data(inputpath)
        with open("{}.pickle".format(inputpath), "rb") as r:
            data = pickle.load(r)

        info = data["info"]
        questions = data["questions"]
        answers = data["answers"]

        return info, questions, answers
            
    except FileNotFoundError:

        print("EMBEDDING INPUTS...")

        data = get_data(inputpath)   
        embed = hub.KerasLayer(module_url)
        embed.trainable = False

        info, questions, answers = [], [], []
        print("Embedding Data...")
        for row in data:
            sentences, question, answer = row
            #print(sentences)
            #print(question)
            #print(answer)

            s = list(itertools.chain(*sentences))
            while(len(s) < get_data.max_slen):
                s.append("<STOP>")
            info.append(embed(s))

            q = list(itertools.chain(question))
            while(len(q) < get_data.max_qlen):
                q.append("<STOP>")
            
            questions.append(embed(q))
            answers.append(embed([answer]))

        info = tf.convert_to_tensor(info)
        questions = tf.convert_to_tensor(questions)
        answers = tf.convert_to_tensor(answers)

        print(info.shape, questions.shape, answers.shape)

        with open("{}.pickle".format(inputpath), "wb") as w:
            pickle.dump({"info": info, "questions": questions, "answers": answers}, w)

        return info, questions, answers






