import pandas as pd
import csv
from util import extract_conversations, get_qa_list, build_word_dict, get_init_embedding
import random


# get train, dev, and test sets indices
def get_indices(indices, train_ratio=0.6, dev_ratio=0.2):
    random.shuffle(indices)
    length = len(indices)
    train_indices = indices[:int(length * train_ratio)]
    dev_indices = indices[int(length * train_ratio):int(length * (train_ratio + dev_ratio))]
    test_indices = indices[int(length * (train_ratio + dev_ratio)):]
    return train_indices, dev_indices, test_indices


def separate_dataset(qa_list, train=0.6, dev=0.2):
    train_indices, dev_indices, test_indices = get_indices(list(range(len(qa_list))), train_ratio=train, dev_ratio=dev)
    train, dev, test =  [qa_list[i] for i in train_indices], [qa_list[i] for i in dev_indices], [qa_list[i] for i in test_indices]
    save_dataset(train, "data/train")
    save_dataset(dev, "data/dev")
    save_dataset(test, "data/test")
    return train, dev, test

def save_dataset(dataset, path):
    with open(path, "w+") as f:
        for qa in dataset:
            f.write("\t".join(qa))
            f.write("\n")


if __name__ == "__main__":
    input_files = [
        "data/export_2018-07-04_train.json",
        "data/export_2018-07-05_train.json",
        "data/export_2018-07-06_train.json",
        "data/export_2018-07-07_train.json",
    ]

    conversations = extract_conversations(input_files)
    print(conversations[5])
    qa_list = get_qa_list(conversations)
    print(len(qa_list))
    train, dev, test = separate_dataset(qa_list, train=0.85, dev=0.15)
    print(len(train), len(dev), len(test))
