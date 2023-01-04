import random
import argparse
import torch
import os
import sys

from model.get_model import get_model
from data.load_data import get_data, process_abstracts, get_inputs, AbstractsDataset
from train.trainer import train_model, eval_model

def main(args):
    pretrained_model_path = args.pretrained_model_path
    batch_size= args.batch_size
    epochs = args.epochs
    start_year = args.start_year
    end_year = args.end_year

    model = get_model(pretrained_model_path)



    for y in range(start_year, end_year+1):
        base_data_path = "/grand/projects/SuperBERT/mansisak/bert-abstracts/"
        year = str(y)
        #subject = "Philosophy"
        subject = "Computer-Science"
        data_dir_path = base_data_path + year + "/" + subject + "/"

        #TODO (MS): load checkpointed model by subject and year!!
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

                model = train_model(epochs, model, loader) 

        #Create Checkpoint by subject and year
        checkpoints_path = '/grand/projects/SuperBERT/mansisak/semantic_decay_models/checkpoints/'
        checkpoint_dir_path = checkpoints_path + year + "/" + subject + "/"
        model.save_pretrained(checkpoint_dir_path)

    base_data_path = "/grand/projects/SuperBERT/mansisak/bert-abstracts/"
    year = str(1931)
    subject = "Computer-Science"

    eval_model(model, batch_size, base_data_path, year, subject)
    """

    base_data_path = "/grand/projects/SuperBERT/mansisak/bert-abstracts/"
    year = str(1932)
    subject = "Biology"

    eval_model(model, base_data_path, year, subject)

    """
    return 0

#Set Arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Semantic Decay NSP Fine Tuning Arguments')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=1, help='Epochs')
    parser.add_argument('--start_year', type=int, default=1931, help='Year for which to begin training')
    parser.add_argument('--end_year', type=int, default=1932, help='Year for which to end training')
    parser.add_argument('--pretrained_model_path', type=str, default='/grand/projects/SuperBERT/mansisak/semantic_decay_models/bookcorpus_pretraining/'
            , help='Path to pretrained model weights (or valid name of hugging face pretrained model)')

args = parser.parse_args()

main(args)
