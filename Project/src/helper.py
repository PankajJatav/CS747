import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from copy import deepcopy
from DMN import DMN

flatten = lambda l: [item for sublist in l for item in sublist]
random.seed(1024)

USE_CUDA = torch.cuda.is_available()
gpus = [0]
torch.cuda.set_device(gpus[0])

FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor

torch.__version__

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex: eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch

    if eindex >= len(train_data):
        batch = train_data[sindex:]
        yield batch

def pad_to_batch(batch, w_to_ix):
    fact, q, a = list(zip(*batch))
    max_fact = max([len(f) for f in fact])
    max_len = max([f.size(1) for f in flatten(fact)])
    max_q = max([qq.size(1) for qq in q])
    max_a = max([aa.size(1) for aa in a])

    facts, fact_masks, q_p, a_p = [], [], [], []
    for i in range(len(batch)):
        fact_p_t = []
        for j in range(len(fact[i])):
            if fact[i][j].size(1) < max_len:
                fact_p_t.append(torch.cat(
                    [fact[i][j], Variable(LongTensor([w_to_ix['<PAD>']] * (max_len - fact[i][j].size(1)))).view(1, -1)],
                    1))
            else:
                fact_p_t.append(fact[i][j])

        while len(fact_p_t) < max_fact:
            fact_p_t.append(Variable(LongTensor([w_to_ix['<PAD>']] * max_len)).view(1, -1))

        fact_p_t = torch.cat(fact_p_t)
        facts.append(fact_p_t)
        fact_masks.append(torch.cat(
            [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in fact_p_t]).view(
            fact_p_t.size(0), -1))

        if q[i].size(1) < max_q:
            q_p.append(
                torch.cat([q[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_q - q[i].size(1)))).view(1, -1)], 1))
        else:
            q_p.append(q[i])

        if a[i].size(1) < max_a:
            a_p.append(
                torch.cat([a[i], Variable(LongTensor([w_to_ix['<PAD>']] * (max_a - a[i].size(1)))).view(1, -1)], 1))
        else:
            a_p.append(a[i])

    questions = torch.cat(q_p)
    answers = torch.cat(a_p)
    question_masks = torch.cat(
        [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in questions]).view(
        questions.size(0), -1)

    return facts, fact_masks, questions, question_masks, answers

def prepare_sequence(seq, to_index):
    idxs = list(map(lambda w: to_index[w] if to_index.get(w) is not None else to_index["<UNK>"], seq))
    return Variable(LongTensor(idxs))

def pad_to_fact(fact, x_to_ix):  # this is for inference

    max_x = max([s.size(1) for s in fact])
    x_p = []
    for i in range(len(fact)):
        if fact[i].size(1) < max_x:
            x_p.append(
                torch.cat([fact[i], Variable(LongTensor([x_to_ix['<PAD>']] * (max_x - fact[i].size(1)))).view(1, -1)],
                          1))
        else:
            x_p.append(fact[i])

    fact = torch.cat(x_p)
    fact_mask = torch.cat(
        [Variable(ByteTensor(tuple(map(lambda s: s == 0, t.data))), volatile=False) for t in fact]).view(fact.size(0),
                                                                                                         -1)
    return fact, fact_mask

def preprocessing(file_loc):
    file_p = open(file_loc).readlines()
    file = [d[:-1] for d in file_p]
    data = []
    fact = []
    qa = []
    for d in file:
        index = d.split(' ')[0]
        if (index == '1'):
            fact = []
            qa = []
        if ('?' in d):
            temp = d.split('\t')
            ques = temp[0].strip().replace('?', '').split(' ')[1:] + ['?']
            ans = temp[1].split() + ['</s>']
            temp_s = deepcopy(fact)
            data.append([temp_s, ques, ans])
        else:
            fact.append(d.replace('.', '').split(' ')[1:] + ['</s>'])
    return data


def train_and_validate(train_file_loc, test_file_loc):

    HIDDEN_SIZE = 80
    BATCH_SIZE = 64
    LR = 0.001
    EPOCH = 70
    NUM_EPISODE = 3
    EARLY_STOPPING = False

    train_data = preprocessing(train_file_loc)

    fact, q, a = list(zip(*train_data))
    vocab = list(set(flatten(flatten(fact)) + flatten(q) + flatten(a)))

    word_to_index = {'<PAD>': 0, '<UNK>': 1, '<s>': 2, '</s>': 3}
    for vo in vocab:
        if word_to_index.get(vo) is None:
            word_to_index[vo] = len(word_to_index)
    index_to_word = {v: k for k, v in word_to_index.items()}

    for s in train_data:
        for i, fact in enumerate(s[0]):
            s[0][i] = prepare_sequence(fact, word_to_index).view(1, -1)
        s[1] = prepare_sequence(s[1], word_to_index).view(1, -1)
        s[2] = prepare_sequence(s[2], word_to_index).view(1, -1)

    model = DMN(len(word_to_index), HIDDEN_SIZE, len(word_to_index))
    model.init_weight()
    if USE_CUDA:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCH):
        losses = []
        if EARLY_STOPPING:
            break

        for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
            facts, fact_masks, questions, question_masks, answers = pad_to_batch(batch, word_to_index)
            model.zero_grad()
            pred = model(facts, fact_masks, questions, question_masks, word_to_index, answers.size(1), NUM_EPISODE, True)
            loss = loss_function(pred, answers.view(-1))
            losses.append(loss.data.tolist())

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("[%d/%d] mean_loss : %0.2f" % (epoch, EPOCH, np.mean(losses)))

                if np.mean(losses) < 0.01:
                    EARLY_STOPPING = True
                    print("Early Stopping!")
                    break
                losses = []

    torch.save(model, 'DMN_'+ train_file_loc.split('/')[2].replace('_train.txt', '') +'.pkl')

    test_data = preprocessing(test_file_loc)

    for t in test_data:
        for i, fact in enumerate(t[0]):
            t[0][i] = prepare_sequence(fact, word_to_index).view(1, -1)
        t[1] = prepare_sequence(t[1], word_to_index).view(1, -1)
        t[2] = prepare_sequence(t[2], word_to_index).view(1, -1)

    accuracy = 0
    for t in test_data:
        fact, fact_mask = pad_to_fact(t[0], word_to_index)
        question = t[1]
        question_mask = Variable(ByteTensor([0] * t[1].size(1)), requires_grad=False).unsqueeze(0)
        answer = t[2].squeeze(0)

        model.zero_grad()
        pred = model([fact], [fact_mask], question, question_mask, word_to_index, answer.size(0), NUM_EPISODE, False)
        if pred.max(1)[1].data.tolist() == answer.data.tolist():
            accuracy += 1


    print(accuracy / len(test_data) * 100)

    try:

        t = random.choice(test_data)
        fact, fact_mask = pad_to_fact(t[0], word_to_index)
        question = t[1]
        question_mask = Variable(ByteTensor([0] * t[1].size(1)), requires_grad=False).unsqueeze(0)
        answer = t[2].squeeze(0)

        model.zero_grad()
        pred = model([fact], [fact_mask], question, question_mask, answer.size(0), NUM_EPISODE, False)

        print("Facts : ")
        print('\n'.join([' '.join(list(map(lambda x: index_to_word[x], f))) for f in fact.data.tolist()]))
        print("")
        print("Question : ", ' '.join(list(map(lambda x: index_to_word[x], question.data.tolist()[0]))))
        print("")
        print("Answer : ", ' '.join(list(map(lambda x: index_to_word[x], answer.data.tolist()))))
        print("Prediction : ", ' '.join(list(map(lambda x: index_to_word[x], pred.max(1)[1].data.tolist()))))
    except:
        print("Error occurred")
        return accuracy / len(test_data) * 100

    return accuracy / len(test_data) * 100


from tabulate import tabulate

data_path = 'data/en-10k/'
import glob

all_train_files = glob.glob(data_path + "*train.txt")
results = []
for file in all_train_files:
    train_file_loc = file
    test_file_loc = file.replace('train', 'test')
    accuracy = train_and_validate(train_file_loc, test_file_loc)
    results.append([file.replace('_train.txt', ''), accuracy] )
print(tabulate(results))
