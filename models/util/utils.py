import os, sys
from datetime import datetime

def pretty_time(seconds, granularity=2):
        intervals = (('w', 604800),('d', 86400),('h', 3600),('m', 60),('s', 1))
        result = []    
        seconds = int(seconds)    
        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count            
                result.append("{}{}".format(value, name))
        return ':'.join(result[:granularity])

def clean_sequences(sequences, lookup):
    """
        Cleans BOS and EOS from sequences.
        sequences (list): is a list of lists containing ints corresponding to the lookup
    """
    bos_id = lookup.convert_tokens_to_ids(lookup.bos_token)
    eos_id = lookup.convert_tokens_to_ids(lookup.eos_token)
    cleaned_sequences = []        
    for seq in sequences:
        lst = []
        for i, value in enumerate(seq):                                
            if i == 0 and value == bos_id: # skip bos
                continue
            if i>0 and value == eos_id: # stop before first eos       
                break
            lst.append(value)
        cleaned_sequences.append(lst)
    
    return cleaned_sequences
    
def select_processing_device(gpu_id = None, verbose = False):
    def _get_freer_gpu():
        try:    
            import numpy as np
            import subprocess
            os_string = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True).decode("utf-8").strip().split("\n")    
            memory_available = [int(x.strip().split()[2]) for x in os_string]
            return int(np.argmax(memory_available))
        except:
            print("Warning: Could not execute 'nvidia-smi' on this platform, selecting default GPU id = 0")
            return 0

    import torch
    if torch.cuda.is_available():
        if gpu_id is None: # auto select GPU
            freer_gpu = _get_freer_gpu()
            if verbose:
                print("Auto-selecting CUDA device #{}: {}".format(freer_gpu, torch.cuda.get_device_name(freer_gpu)))
            torch.cuda.set_device(freer_gpu)
        else:
            if not isinstance(gpu_id, int):
                raise Exception("ERROR: Please specify the GPU id as a valid integer!")
            if verbose:
                print("Selected CUDA device #{}: {}".format(gpu_id, torch.cuda.get_device_name(gpu_id)))
            torch.cuda.set_device(gpu_id)
        return torch.device('cuda')
    else:
        return torch.device('cpu')