"""
This script creates a vocabulary based on the input folder.

Set input parameters below:
"""
arg = {}
arg["input_folder"] = "../../data/processed" # where the cnn and dm folders contain the processed jsons
arg["output_folder"] = "../../train/transformer" # where to store the vocab dict and indexes
arg["lowercase"] = True # whether to lowercase or not
arg["minimum_frequency"] = 5 # minimum frequency for a word to be kept in the vocab
arg["max_sequence_len"] = 400 # max length of an instance
arg["validation_fraction"] = 0.05 # fraction to use as validation
arg["test_fraction"] = 0.05 # fraction to test on
arg["full_data_fraction"] = 0.1 # what fraction from all avaliable data to use (1.0 if you want full dataset)
arg["x_field"] = "x_tokenized_original_sentence"
arg["y_field"] = "y_tokenized_original_sentences"
# ######################################


import os, sys, json
import torch

print("Parameters: ")
print(arg)

# create output folder
if not os.path.exists(arg["output_folder"]):
        os.makedirs(arg["output_folder"])

# first step, create dictionary 





