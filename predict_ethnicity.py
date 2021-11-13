
import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import argparse
import numpy as np
import json
import pandas as  pd
import string
from typing import Union
import os
import unicodedata
import re
import sys



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, class_amount: int=0, embedding_size: int=64, hidden_size: int=10, layers: int=1, dropout_chance: float=0.5, kernel_size: int=3, channels: list=[32, 64, 128]):
        super(Model, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dropout_chance = dropout_chance

        self.kernel_size = kernel_size
        self.channels = channels

        self.embedder = nn.Embedding(29, self.embedding_size)

        self.conv1 = nn.Sequential(nn.Conv1d(self.embedding_size, self.channels[0], kernel_size=self.kernel_size),
                                   nn.ReLU())
        
        self.lstm = nn.LSTM(input_size=self.channels[-1], hidden_size=self.hidden_size, num_layers=self.layers, batch_first=True)
        
        self.dropout = nn.Dropout2d(p=self.dropout_chance)
        self.linear1 = nn.Linear(self.hidden_size, class_amount)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedder(x.type(torch.LongTensor).to(device=device))
        x = x.squeeze(2).transpose(1, 2)
        
        x = self.conv1(x)
        x = x.transpose(1, 2)

        hidden = (torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device), torch.zeros(self.layers, x.size(0), self.hidden_size).to(device=device))
        x, _ = self.lstm(x)
        x = x[:, -1]

        x = self.dropout(x)

        x = self.linear1(x)
        x = self.logSoftmax(x)

        return x


def get_flags() -> Union[list, str, str, int, str]:
    """ handles console arguments

    :return list: list of names to predict ethnicities
    :return str: path of csv-file in which to save ethnicities
    :return str: model configuration name
    :return int: batch-size for forward pass
    :return str: host device for the model
    """

    parser = argparse.ArgumentParser()

    csv_names_group = parser.add_argument_group("classify multiple names")
    single_name_group = parser.add_argument_group("classify single name")

    csv_names_group.add_argument("-i", "--input", required=False, help="path to .csv containing (first and last) names; must contain one column called 'names' (name freely selectable)")
    csv_names_group.add_argument("-o", "--output", required=False, help="path to .csv in which the names along with the predictions will be stores (file will be created if it doesn't exist; name freely selectable)")
    csv_names_group.add_argument("-d", "--device", required=False, help="must be either 'gpu' or 'cpu' (standard: 'gpu' if cuda support is detected, else 'cpu')")
    csv_names_group.add_argument("-b", "--batchsize", required=False, help="specifies how many names will be processed in parallel (standard: process all names in parallel; if it crashes choose a batch-size smaller than the amount of names in your .csv file; the bigger the batchsize the faster it will classify the names)")
    single_name_group.add_argument("-n", "--name", required=False, help="first and last name (upper-/ lower case doesn't matter)")
    parser.add_argument("-m", "--model", required=False, help="folder name of model configuration which can be chosen from 'model_configurations/' (standard: '21_nationalities_and_else')")

    args = vars(parser.parse_args())

    # check if -/--name is used and -i/--input not
    if args["name"] != None and args["input"] == None:
        names = [args["name"]]
        csv_out_path = None
    
    # check if -/--name is not used but -i/--input is
    elif args["name"] == None and args["input"] != None:
        csv_in_path = args["input"]
        csv_out_path = args["output"]
        names = pd.read_csv(csv_in_path)["names"].tolist()

    # check if -/--name and -c/--csv are both not used (raise error)
    elif args["name"] == None and args["input"] == None:
        raise ValueError("Either -n/--name or -i/--input must be set!")

    # check if -/--name and -c/--csv are both used (raise error)
    elif args["name"] != None and args["input"] != None:
        raise ValueError("-n/--name and -i/--input can't both be set!")

    if args["input"] != None and args["output"] == None or args["input"] == None and args["output"] != None:
        raise ValueError("When using -i/--input the -o/--output flag is required (and the other way around)!")

    # get model
    if args["model"] == None:
        model_config_folder = "model_configurations/21_nationalities_and_else"
    elif os.path.exists("model_configurations/" + args["model"]):
        model_config_folder = "model_configurations/" + args["model"]
    else:
        raise FileNotFoundError("The given model configuration folder does not exist!")

    # get batch-size
    if args["batchsize"] == None or int(args["batchsize"]) > len(names):
        batch_size = len(names)
    else:
        batch_size = int(args["batchsize"])

    # get device
    if args["device"] == None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args["device"].lower() == "cpu":
        device = torch.device("cpu")
    elif args["device"].lower() == "gpu":
        if not torch.cuda.is_available():
            print("Couldn't find cuda on your system! Please use 'CPU' or install cuda when possible! Proceeding with CPU...")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        raise NameError("Please use either 'GPU' or 'CPU' as device type!")

    return names, csv_out_path, model_config_folder, batch_size, device


def replace_special_chars(name: str) -> str:
    """ replaces all apostrophe letters with their base letters and removes all other special characters incl. numbers
    
    :param str name: name
    :return str: normalized name
    """

    name = u"{}".format(name)
    name = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("utf-8")
    name = re.sub("[^A-Za-z -]+", "", name)

    return name


def preprocess_names(names: list=[str], batch_size: int=128) -> torch.tensor:
    """ create a pytorch-usable input-batch from a list of string-names
    
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
    

def predict(input_batch: torch.tensor, model_config: dict) -> str:
    """ load model and predict preprocessed name

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

        # convert numerics to country name
        predicted_ethnicites = []
        for idx in range(len(predictions)):
            prediction = predictions.cpu().detach().numpy()[idx]
            prediction_idx = list(prediction).index(max(prediction))
            ethnicity = list(classes.keys())[list(classes.values()).index(prediction_idx)]
            predicted_ethnicites.append(ethnicity)

        total_predicted_ethncitities += predicted_ethnicites
    
    return total_predicted_ethncitities
    

if __name__ == "__main__":
    # get names from console arguments
    names, csv_out_path, model_config_folder, batch_size, device = get_flags()

    # get model configuration
    with open(model_config_folder + "/nationalities.json", "r") as f: classes = json.load(f)
    with open(model_config_folder + "/config.json", "r") as f: model_parameter_config = json.load(f)
    model_file = model_config_folder + "/model.pt"

    # preprocess inputs
    input_batch = preprocess_names(names=names, batch_size=batch_size)
    
    model_config = {
        "model-file": model_file,
        "amount-classes": len(classes),
        "embedding-size": model_parameter_config["embedding-size"],
        "hidden-size": model_parameter_config["hidden-size"],
        "rnn-layers": model_parameter_config["rnn-layers"],
        "cnn-parameters": model_parameter_config["cnn-parameters"]
    }

    # predict ethnicities
    ethnicities = predict(input_batch, model_config)

    # check if the -i/--input and -o/--output flag was set, by checking if there is a csv-save-file, if so: save names with their ethnicities
    if csv_out_path != None:
        df = pd.DataFrame()
        df["names"] = names
        df["ethnicities"] = ethnicities

        open(csv_out_path, "w+").close()
        df.to_csv(csv_out_path, index=False)
    
        print("\nClassified all names and saved to {} .\n".format(csv_out_path))
    
    # if a single name was parsed using -n/--name, print the predicition
    else:
        print("\nname: {} - predicted ethnicity: {}".format(names[0], ethnicities[0]))

