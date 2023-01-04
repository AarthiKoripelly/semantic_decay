import torch
from tqdm import tqdm  # for our progress bar
import os
import sys

sys.path.append('../')
from data.load_data import get_data, process_abstracts, get_inputs, AbstractsDataset


def train_model(epochs, model, batch_size, base_data_path, year, subject):
    base_data_path = base_data_path
    year = year
    subject = subject
    data_dir_path = base_data_path + year + "/" + subject + "/"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=5e-6)

    for filename in os.listdir(data_dir_path):
        data_path = os.path.join(data_dir_path, filename)
        # checking if it is a file
        if os.path.isfile(data_path):
            print(data_path)

            bag, bag_size, abstracts = get_data(data_path)
            sentence_a, sentence_b, label = process_abstracts(abstracts, bag, bag_size)
            if(len(sentence_a) <= 0):
                print("DATA LEN: ", len(sentence_a))
                continue
            inputs = get_inputs(sentence_a, sentence_b, label)
            dataset = AbstractsDataset(inputs)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  

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
    return model

def eval_model(model, batch_size, base_data_path, year, subject):
    base_data_path = base_data_path
    year = year
    subject = subject
    data_dir_path = base_data_path + year + "/" + subject + "/"

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate evaluation mode
    model.eval()

    total_pairs = 0
    correct_classifications = 0;
    for filename in os.listdir(data_dir_path):
        data_path = os.path.join(data_dir_path, filename)
        # checking if it is a file
        if os.path.isfile(data_path):
            print(data_path)

            bag, bag_size, abstracts = get_data(data_path)
            sentence_a, sentence_b, label = process_abstracts(abstracts, bag, bag_size)
            if(len(sentence_a) <= 0):
                print("DATA LEN: ", len(sentence_a))
                continue
            inputs = get_inputs(sentence_a, sentence_b, label)
            dataset = AbstractsDataset(inputs)
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


            with torch.no_grad():
                loop = tqdm(loader, leave=True)
                for batch in loop:
                    # initialize calculated gradients (from prev step)
                    # pull all tensor batches required for training
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    # process
                    outputs = model(input_ids, attention_mask=attention_mask,
                                    labels=labels)

                    correct_classifications += labels.shape[0] - torch.sum(torch.abs(torch.argmax(outputs.logits, axis=1) - labels.squeeze())).item()
                    total_pairs += labels.shape[0]
                    accuracy = correct_classifications / total_pairs

                    # print relevant info to progress bar
                    loop.set_description(f'Evaluation')
                    loop.set_postfix(accuracy=accuracy)

    print("accuracy: ", accuracy)
