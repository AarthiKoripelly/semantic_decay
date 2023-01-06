import random
from transformers import BertTokenizer
import torch
from tqdm import tqdm  # for our progress bar
import os
import sys

#Need to seed everything to ensure reproducibility
random.seed(42)

#Get tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_data(data_path):
    with open(data_path, "r") as fp:
      abstracts = fp.read().split('\n\n')

    bag = [item for abstract in abstracts for item in abstract.split('\n') if item != '']
    bag_size = len(bag)
    return bag, bag_size, abstracts

def process_abstracts(abstracts, bag, bag_size):
    sentence_a = []
    sentence_b = []
    label = []
    total_sentences = 0
    for abstract in abstracts:
        sentences = [
            sentence for sentence in abstract.split('\n') if sentence != ''
        ]
        num_sentences = len(sentences)
        total_sentences += num_sentences
        if num_sentences > 1:
            #TODO: this is wasting data, make sure every sentence in each abstract is used
            start = random.randint(0, num_sentences-2)
            # 50/50 whether is IsNextSentence or NotNextSentence
            if random.random() >= 0.5:
                # this is IsNextSentence
                sentence_a.append(sentences[start])
                sentence_b.append(sentences[start+1])
                label.append(0)
            else:
                index = random.randint(0, bag_size-1)
                # this is NotNextSentence
                #TODO: make sure this is actually non-sequential sentences!!
                sentence_a.append(sentences[start])
                sentence_b.append(bag[index])
                label.append(1)
    return sentence_a, sentence_b, label


def get_inputs(sentence_a, sentence_b, label):
    inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    inputs['labels'] = torch.LongTensor([label]).T
    return inputs


class AbstractsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
