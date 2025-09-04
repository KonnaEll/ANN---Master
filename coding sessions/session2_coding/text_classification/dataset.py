from config import *
import pandas as pd
import torch
import random
random.seed(0)
import re


def tsv_to_list(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    texts = df[df.columns[0]].values.tolist()
    df[df.columns[1]] = df[df.columns[1]].replace(labels_mapper)
    labels = df[df.columns[1]].values.tolist()
    return texts, labels


def data_split(texts, labels, split_ratio=0.8):
    # split train data into train and validation
    tuples = list(zip(texts, labels))
    random.shuffle(tuples)
    data_length = len(tuples)
    tuples_train = tuples[:int(split_ratio*data_length)]
    tuples_val = tuples[int(split_ratio*data_length):]
    train_text, train_labels = zip(*tuples_train)
    val_text, val_labels = zip(*tuples_val)
    return train_text, train_labels, val_text, val_labels


# You can skip this part for now if you do not have an idea
def step1(texts):
    # hint: cleaning
    pass


def step2(texts):
    # hint: having a look-up-table using python dictionary
    pass


def step3(texts, vocabs):
    # hint: converting strings to the required data format for an MLP (words --> numbers)
    pass


def step4():
    # we need equal size for all our inputs
    pass


def create_dataset(texts, labels, vocabs=None, max_length=50):
    # call all defined functions in sequence

    # step1()
    # .....
    # vectors, labels = step4()

    vectors = torch.Tensor(vectors)
    labels = torch.Tensor(labels)
    return vectors, labels, vocabs


if __name__ == "__main__":
    texts, labels = tsv_to_list(train_path)
    texts_train, labels_train, texts_val, labels_val = data_split(texts, labels, split_ratio=0.8)

    # print out each function to see the resulted outputs
    # step1()

    # step2()

    # step3()

    # uncomment below lines after completing all functions
    # texts, labels, vocabs = create_dataset(texts_train, labels_train, max_length=max_length)
    # print(texts.shape)
    # print(labels.shape)
