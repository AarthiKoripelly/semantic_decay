from transformers import BertConfig
config = BertConfig.from_pretrained("bert-base-uncased")

from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#This will instantiatiate a pre-trained model for NSP
#TODO (MS): load a given pretrained model
model = BertForNextSentencePrediction.from_pretrained('/grand/projects/SuperBERT/mansisak/semantic_decay_models/bookcorpus_pretraining/')

#TODO (MS): fix path to data
file_path = "/grand/projects/SuperBERT/mansisak/bert-abstracts/2008/Biology/20220701_070950_00030_vrytk_d58b1dad-8368-42de-b191-b9a2da885938.txt"
with open(file_path, "r") as fp:
  abstracts = fp.read().split('\n\n')

print("Opened Abstracts")

bag = [item for abstract in abstracts for item in abstract.split('\n') if item != '']
bag_size = len(bag)


import random

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

print("Finished Processing All Abstracts")

inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

inputs['labels'] = torch.LongTensor([label]).T

class AbstractsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = AbstractsDataset(inputs)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)

# activate training mode
model.train()
# initialize optimizer
optim = torch.optim.AdamW(model.parameters(), lr=5e-6)

from tqdm import tqdm  # for our progress bar

epochs = 1

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

#TODO: need to evaluate model accuracy
model.eval()

total_pairs = 0
correct_classifications = 0;
with torch.no_grad():
    for batch in tqdm(loader, leave=True):
        # initialize calculated gradients (from prev step)
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # extract loss
        loss = outputs.loss

        correct_classifications += labels.shape[0] - torch.sum(torch.abs(torch.argmax(outputs.logits, axis=1) - labels.squeeze())).item()
        total_pairs += labels.shape[0]

        # print relevant info to progress bar
        loop.set_description(f'Evaluation')
        loop.set_postfix(accuracy=correct_classifications/total_pairs)
        #loop.set_postfix(loss=loss.item())

accuracy = correct_classifications / total_pairs
print("accuracy: ", accuracy)

#TODO: 
