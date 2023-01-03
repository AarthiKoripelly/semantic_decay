from transformers import BertForNextSentencePrediction

#This will instantiatiate a pre-trained model for NSP
def get_model(model_path):
    model = BertForNextSentencePrediction.from_pretrained(model_path)
    return model
