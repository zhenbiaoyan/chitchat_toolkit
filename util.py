from nltk.tokenize import word_tokenize
import collections
import pickle
import json
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
import os


# build a batch iterator
def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


# build word2index and index2word dictionaries
def build_word_dict(step, path):
    words = get_words_list(path)
    word_dict = {}
    if step == "train":
        word_counter = collections.Counter(words).most_common()
        word_dict = {
            "<padding>": 0,
            "<unk>": 1,
            "<s>": 2,
            "</s>": 3,
        }
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "dev":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    # build reversed index dict
    reversed_dict = {}
    for w in word_dict:
        reversed_dict[word_dict[w]] = w

    return word_dict, reversed_dict


# turn conversations to q&a
def get_qa_list(conversations):
    qa_list = []
    for conversation in conversations:
        for i in range(len(conversation) - 1):
            qa_list.append([conversation[i], conversation[i + 1]])

    return qa_list

def get_words_list(path):
    qa_list = load_qa_list(path)
    words = []
    for qa in qa_list:
        for sentence in qa:
            for word in word_tokenize(sentence):
                words.append(word)

    return words

def load_qa_list(path):
    qa_list = []
    with open(path) as f:
        for line in f:
            qa_list.append(line.split('\t'))

    return qa_list


# generate word index matrix
def build_dataset(step, qa_list, word2index, question_max_len=30, answer_max_len=30):

    if step not in ('train', 'dev'):
        raise NotImplementedError
    question_list, answer_list = [], []
    try:

        # question_list, answer_list = [qa[0] for qa in qa_list], [qa[1] for qa in qa_list]
        for qa in qa_list:
            question_list.append(qa[0])
            answer_list.append(qa[1])
    except:
        print(qa)


    x = [word_tokenize(q)[:question_max_len] for q in question_list]
    x = [[word2index.get(w, word2index["<unk>"]) for w in q] for q in x]
    x = [d + (question_max_len - len(d)) * [word2index["<padding>"]] for d in x]

    if step == "dev":
        return x, []
    else:
        y = [word_tokenize(a)[:answer_max_len] for a in answer_list]
        y = [[word2index.get(w, word2index["<unk>"]) for w in a] for a in y]
        return x, y


# get fixed sized embedding vectors for each sentence
def extract_conversations(files):
    result = []

    for f in files:
        with open(f) as json_file:
            data = json.load(json_file)
            for p in data:
                dialog = p['dialog']
                sub_result = []
                if not dialog or len(dialog) < 2:
                    continue
                for message in dialog:
                    text = clean(message['text'])
                    sub_result.append(text)
                result.append(sub_result)

    return result

def clean(sentence):
    return sentence.replace('\n', ' ').replace('\t', ' ')

def get_init_embedding(index2word, embedding_size=50):
    glove_file = "glove/glove.6B.300d.txt"
    word2vec_file = get_tmpfile(os.path.join(os.path.dirname(__file__), "glove/word2vec_format.vec"))
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = []
    for i in range(len(index2word)):
        word = index2word[i]
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    return np.array(word_vec_list)

def get_feed_dict(model, word2index, answer_max_len, batch_x, batch_y):
    batch_x_len = list(map(lambda x: len([xx for xx in x if xx != 0]), batch_x))
    batch_decoder_input = list(map(lambda y: [word2index["<s>"]] + y[:49], batch_y))
    batch_decoder_len = list(map(lambda y: len([yy for yy in y if yy != 0]), batch_decoder_input))
    batch_decoder_output = list(map(lambda y: list(y)[:49] + [word2index["</s>"]], batch_y))

    batch_decoder_input = list(
        map(lambda d: d + (answer_max_len - len(d)) * [word2index["<padding>"]], batch_decoder_input))
    batch_decoder_output = list(
        map(lambda d: d + (answer_max_len - len(d)) * [word2index["<padding>"]], batch_decoder_output))

    feed_dict = {
        model.batch_size: len(batch_x),
        model.X: batch_x,
        model.X_len: batch_x_len,
        model.decoder_input: batch_decoder_input,
        model.decoder_len: batch_decoder_len,
        model.decoder_target: batch_decoder_output
    }

    return feed_dict

