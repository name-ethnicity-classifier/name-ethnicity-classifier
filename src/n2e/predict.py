import argparse
import json
import os
import re
import string
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence

from n2e.config import get_model_folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, class_amount: int=0, embedding_size: int=64, hidden_size: int=10, layers: int=1, kernel_size: int=3, channels: list=[32, 64, 128]):
        super(Model, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.kernel_size = kernel_size
        self.channels = channels

        self.embedder = nn.Embedding(29, self.embedding_size)

        self.conv1 = nn.Sequential(nn.Conv1d(self.embedding_size, self.channels[0], kernel_size=self.kernel_size), nn.ReLU())
        self.lstm = nn.LSTM(input_size=self.channels[-1], hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_size, class_amount)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedder(x.type(torch.LongTensor).to(device=device))
        x = x.squeeze(2).transpose(1, 2)
        
        x = self.conv1(x)
        x = x.transpose(1, 2)

        x, _ = self.lstm(x)
        x = x[:, -1]

        x = self.linear1(x)
        x = self.logSoftmax(x)

        return x

def replace_special_chars(name: str) -> str:
    """
    Replaces all apostrophe letters with their base letters and removes all other special characters incl. numbers
    
    :param str name: name
    :return str: normalized name
    """

    name = u"{}".format(name)
    name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub("[^A-Za-z -]+", "", name)

    return name


def preprocess_names(names: list=[str], batch_size: int=128) -> torch.tensor:
    """
    Creates a pytorch compatible input-batch from a list of string-names
    
    :param list names: list of names (strings)
    :param int batch_size: batch-size for the forward pass
    :return torch.tensor: preprocessed names (to tensors, padded, encoded)
    """

    sample_batch = []
    for name in names:
        try:
            # remove special characters
            name = replace_special_chars(name)

            # create index-representation from string name, ie: "joe" -> [10, 15, 5], indices go from 1 ("a") to 28 ("-")
            alphabet = list(string.ascii_lowercase.strip()) + [" ", "-"]
            int_name = []
            for char in name:
                int_name.append(alphabet.index(char.lower()) + 1)
            
            name = torch.tensor(int_name)
            sample_batch.append(name)

        except:
            raise ValueError("\nCould not process the name: '{}'! Aborting.".format(name))

    padded_batch = pad_sequence(sample_batch, batch_first=True)

    padded_to = list(padded_batch.size())[1]
    padded_batch = padded_batch.reshape(len(sample_batch), padded_to, 1).to(device=device)

    if padded_batch.shape[0] == 1 or batch_size == padded_batch.shape[0]:
        padded_batch = padded_batch.unsqueeze(0)
    else:
        padded_batch = torch.split(padded_batch, batch_size)

    return padded_batch


def get_ethnicity_predictions(predictions: np.array, classes: list) -> list[str]:
    """
    Collects the highest confidence ethnicity for every prediction in a batch.
    For example if the model classified a batch of two names into eithher "german" or "greek":
    > [(german, 0.9), (greek, 0.8)]

    :param predictions: The output predictions of the model
    :param classes: A list containing all the classes which a model can classify
    :return: A list containing the predicted ethnicity and confidence score for each name
    """

    predicted_ethnicites = []
    for prediction in predictions:
        prediction_idx = list(prediction).index(max(prediction))
        ethnicity = classes[prediction_idx]
        predicted_ethnicites.append((ethnicity, round(100 * float(np.exp(max(prediction))), 3)))

    return predicted_ethnicites


def get_ethnicity_distributions(predictions: np.array, classes: list) -> list[dict]:
    """
    Collects the entire output distribution for every predictions in a batch
    For example if the model classified a batch of two names into either "german" or "greek":
    > [{german: 0.9, greek: 0.1}, {german: 0.2, greek: 0.8}]

    :param predictions: The output predictions of the model
    :param classes: A list containing all the classes which a model can classify
    :return: A list containing an output distribution for each name
    """

    predicted_ethnicites = []

    for prediction in predictions:
        ethnicity_distribution = {}
        for idx, ethnicity in enumerate(classes):
            confidence = round(100 * float(np.exp(prediction[idx])), 3)
            ethnicity_distribution[ethnicity] = confidence

        predicted_ethnicites.append(ethnicity_distribution)

    return predicted_ethnicites

    

def predict(input_batch: torch.tensor, model_config: dict, classes: list ,get_distribution: bool=False) -> str:
    """ 
    Loads model and predict preprocessed name

    :param torch.tensor input_batch: input-batch
    :param str model_path: path to saved model-paramters
    :param dict classes: a dictionary containing all countries with their class-number
    :return str: predicted ethnicities
    """

    # prepare model (map model-file content from gpu to cpu if necessary)
    model = Model(
        class_amount=model_config["amount-classes"], 
        embedding_size=model_config["embedding-size"],
        hidden_size=model_config["hidden-size"],
        layers=model_config["rnn-layers"],
        kernel_size=model_config["cnn-parameters"][0],
        channels=model_config["cnn-parameters"][1]
    ).to(device=device)

    model_path = model_config["model-file"]

    if device != "cuda:0":
        model.load_state_dict(torch.load(model_path, map_location={"cuda:0": "cpu"}))
    else:
        model.load_state_dict(torch.load(model_path))

    model = model.eval()

    # classify names    
    total_predicted_ethncitities = []

    for batch in input_batch:
        predictions = model(batch.float())

        predictions = model(batch.float()).cpu().detach().numpy()

        if get_distribution:
            # get entire ethnicity confidence distribution for each name
            prediction_result = get_ethnicity_distributions(predictions, classes=classes)
        else:
            # get only the ethnicity with the highest confidence for each name
            prediction_result = get_ethnicity_predictions(predictions, classes=classes)

        total_predicted_ethncitities.extend(prediction_result)

    return total_predicted_ethncitities
    
def predict_ethnicities (names: list[str], batch_size: int=128, model: str="21_nationalities_and_else", get_distribution: bool=False) -> list: 
    """
    Predicts the ethnicity of a given list of names. Configured by arguments

    :param names: list of names to be classified
    :param batch_size: number of names to be classified at a time in batches, defaults to 128
    :param model: model being used, check README for all models, defaults to "21_nationalities_and_else"
    :param get_distribution: If true returns a dict of all coutnry predictions and scores, defaults to False
    :return list: predicted ethincity of each name 
    """

    if isinstance(names, str):
        raise TypeError(f"Names must be provided as a list of strings - did you mean ['{names}']?")

    # preprocess inputs
    input_batch = preprocess_names(names=names, batch_size=batch_size)
    model_config_folder = get_model_folder(model)

    # get model configuration
    with open(model_config_folder + "/classes.json", "r") as f: classes = json.load(f)
    with open(model_config_folder + "/config.json", "r") as f: model_parameter_config = json.load(f)
    model_file = model_config_folder + "/model.pt"
    
    model_config = {
        "model-file": model_file,
        "amount-classes": len(classes),
        "embedding-size": model_parameter_config["embedding-size"],
        "hidden-size": model_parameter_config["hidden-size"],
        "rnn-layers": model_parameter_config["rnn-layers"],
        "cnn-parameters": model_parameter_config["cnn-parameters"]
    }

    # predict ethnicities
    predictions = predict(input_batch, model_config, classes=classes, get_distribution=get_distribution)
    return predictions