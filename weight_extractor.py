import torch
from torch import nn
import os
import numpy as np

def make_directory(base_path : str) -> int :
    """
        Checks if a directory exists and if doesn't creates the directory.

        Args:
        base_path : Directory path which will be created if it doesn't exist.

        Returns 0 if directory exists else 1
    """
    if os.path.exists(base_path) :
        return 0

    # Create the directory since the path doesn't exist.
    os.mkdir(base_path)
    if os.path.exists(base_path) :
        return 0

    # Path doesn't exist as well as directory couldn't be created.
    print("Error : Cannot create desired path : ", base_path)
    return 1

def generate_csv(csv_name : str, weight_matrix : torch.tensor, base_path : str) -> str :
    """
        Generates csv for weights or bias matrix.

        Args:
        csv_name : A string name for csv file which will store the weights.
        weight_matrix : A torch tensor holding weights that will be stored in the matrix.
        base_path : Base path where csv will be stored.
    """
    # Check if base path exists else create directory.
    make_directory(base_path)
    file_path = os.path.join(base_path, csv_name)
    np.savetxt(file_path, weight_matrix.numpy().ravel())
    return file_path

def extract_weights(layer, layer_index, base_path) -> {} :
    """
        Extracts weights, biases and other parameters required to reproduce
        the same output.

        Args:
        layer : An torch.nn object (layer).
        layer_index : A string determining name of csv file that will be appended to
                      name of layer.
                      Eg. if layer = nn.Conv2d and layer_index = 0
                          csv_filename = Conv_layer_index.csv
        base_path : A string depicting base path for storing weight / bias csv.

        Returns dictionary of parameter description and parameters.

        Exceptions:
        Currently this has only been tested for convolutional and batch-norm layer.
    """
    parameter_dictionary = {}
    if isinstance(layer, nn.Conv2d):
        # The layer corresponds to Convolutional layer.
        # For convolution layer we require weights and biases to reproduce the
        # same result.
        parameter_dictionary["name"] = "Convolution2D"
        parameter_dictionary["input-channels"] = layer.in_channels
        parameter_dictionary["output-channels"] = layer.out_channels
        # Assume weight matrix is never empty for nn.Conv2d()
        parameter_dictionary["has-weights"] = 1
        parameter_dictionary["weight-offset"] = 0
        csv_name = "conv_weight_" + layer_index + ".csv"
        parameter_dictionary["weight-csv"] = generate_csv(csv_name, layer.weight.detach(), base_path)
        if layer.bias != None:
            parameter_dictionary["has-bias"] = 1
            parameter_dictionary["bias-offset"] = layer.bias.numel()
            bias_csv_name = "conv_bias_" + layer_index + ".csv"
            parameter_dictionary["bias-csv"] = generate_csv(bias_csv_name, layer.bias.detach(), base_path)
        else:
            parameter_dictionary["has-bias"] = 0
            parameter_dictionary["bias-offset"] = layer.out_channels
            parameter_dictionary["bias-csv"] = "None"
        
    return parameter_dictionary

def parse_model(model, xml_path, base_path, debug : bool) -> int :
    """
        Parses model and generates csv and xml file which will be iterated by C++ translator.

        Args:
        model : PyTorch model for which parameter csv and xml will be created.
        xml_path : Directory where xml with model config will be saved.
        base_path : Directory where csv will be stored.

        Returns 0 if weights are created else return 1.
    """
    layer_index = 0
    for modules in model.features:
        for layer in modules:
            layer_index += 1
            parameter_dict = extract_weights(layer, str(layer_index), base_path)
            if not os.path.exists(parameter_dict["weight-csv"]) and parameter_dict["has-weights"] == 0:
                print("Creating weights failed!")
                return 1
            if debug :
                print("Weights created succesfully for ", parameter_dict.name, " layer index :", layer_index)
    if debug :
        print("Model weights saved! Happy mlpack-translation.")
    return 0