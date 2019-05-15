import os, subprocess
import torch
import torch.nn as nn
from models.util.log import Log
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def get_freer_gpu():  
    try:    
        import numpy as np
        os_string = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True).decode("utf-8").strip().split("\n")    
        memory_available = [int(x.strip().split()[2]) for x in os_string]
        return int(np.argmax(memory_available))
    except:
        print("Warning: Could execute 'nvidia-smi', default GPU selection is id=0")
        return 0


def _print_examples(model, loader, seq_len, src_i2w, tgt_i2w):
    X_sample, y_sample = iter(loader).next()
    seq_len = min(seq_len,len(X_sample))
    X_sample = X_sample[0:seq_len]
    y_sample = y_sample[0:seq_len]
    if model.cuda:
        X_sample = X_sample.cuda()
        y_sample = y_sample.cuda()

    y_pred_dev_sample = model.forward(X_sample, y_sample)
    y_pred_dev_sample = torch.argmax(y_pred_dev_sample, dim=2)

    for i in range(seq_len):
        for j in range(len(X_sample[i])):
            token = str(X_sample[i][j].item())

            if token not in src_i2w.keys():
                print(src_i2w['1'] + " ", end='')
            elif token == '3':
                print(src_i2w['3'], end='')
                break
            else:
                print(src_i2w[token] + " ", end='')
        print()
        for j in range(len(y_sample[i])):
            token = str(y_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()
        for j in range(len(y_pred_dev_sample[i])):
            token = str(y_pred_dev_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'], end='')
                break
            else:
                print(tgt_i2w[token] + " ", end='')
        print()
        print("-" * 40)


#def train(model, epochs, batch_size, lr, n_class, train_loader, valid_loader, test_loader, src_i2w, tgt_i2w, model_path):
def train(model, src_i2w, tgt_i2w, train_loader, valid_loader=None, test_loader=None, model_store_path=None,
          resume=False, max_epochs=100000, patience=10, lr=0.0005):
    if model_store_path is None: # saves model in the same folder as this script
        model_store_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)
    
    log_path = os.path.join(model_store_path,"log")
    log_object = Log(log_path, clear=True)
    print("Working in folder [{}]".format(model_store_path))
    
    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_class = len(tgt_i2w)
    batch_size = len(train_loader.dataset.X[0])
    current_epoch = 0
    current_patience = patience
    best_accuracy = 0.

    if resume: # load checkpoint         
        extra_variables = model.load_checkpoint(model_store_path, extension="best")                
        if "epoch" in extra_variables:
            current_epoch = extra_variables["epoch"]                
        print("Resuming from epoch {}".format(current_epoch))
        load_optimizer_checkpoint (optimizer, model.cuda, model_store_path, extension="best")
    
    while current_patience > 0 and current_epoch < max_epochs:
        print()
        
        # train
        model.train()
        total_loss = 0
        t = tqdm(train_loader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [train]", unit="batches")
        
        for batch_index, (x_batch, y_batch) in enumerate(t):
        #for x_batch, y_batch in train_loader:
            if model.cuda:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            y_in_batch = y_batch[:, :-1]
            y_out_batch = y_batch[:, 1:]

            optimizer.zero_grad()
            
            # x_batch and y_batch shapes: [bs, padded_sequence]
            output = model.forward(x_batch, y_in_batch)
            # output shape: [bs, padded_sequence, n_class]
            #print(output.size())
            #print(output.view(-1, n_class).size())
            #print(y_out_batch.size())
            #input("Asd")
            loss = criterion(output.view(-1, n_class), y_out_batch.contiguous().flatten())                        
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()
            log_average_loss = total_loss / (batch_index+1)
            t.set_postfix(loss=log_average_loss) 

        # dev
        if valid_loader is not None:
            model.eval()            
            dev_accuracy = 0
            
            t = tqdm(valid_loader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="batches")
            for batch_index, (x_dev_batch, y_dev_batch) in enumerate(t):            
                if model.cuda:
                    x_dev_batch = x_dev_batch.cuda()
                    y_dev_batch = y_dev_batch.cuda()

                y_in_dev_batch = y_dev_batch[:, :-1]
                y_out_dev_batch = y_dev_batch[:, 1:]

                y_pred_dev = model.forward(x_dev_batch, y_in_dev_batch).argmax(dim=2).flatten()

                dev_accuracy += accuracy_score(y_out_dev_batch.contiguous().flatten().cpu(), y_pred_dev.cpu())
                
                current_instance_count = batch_index*batch_size + len(x_dev_batch[0]) # previous full batches + current batch
                log_dev_accuracy = dev_accuracy / (current_instance_count/batch_size)
                t.set_postfix(validation_accuracy=log_dev_accuracy)
            
            if log_dev_accuracy > best_accuracy:
                print("\t Best accuracy = {}".format(log_dev_accuracy))
                model.save_checkpoint(model_store_path, extension="best", extra={"epoch":current_epoch})
                save_optimizer_checkpoint (optimizer, model_store_path, extension="best")
                
            _print_examples(model, valid_loader, batch_size, src_i2w, tgt_i2w)
            
        else: # disable patience if no dev provided and always save model 
            current_patience = patience
            model.save_checkpoint(model_store_path, "best", extra={"epoch":current_epoch})
            save_optimizer_checkpoint (optimizer, model_store_path, extension="best")
            
        current_epoch += 1


def save_optimizer_checkpoint (optimizer, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer."+extension)
    #print("Saving optimizer parameters to {} ...".format(filename))    
    torch.save(optimizer.state_dict(), filename)    


def load_optimizer_checkpoint (optimizer, cuda, folder, extension):
    filename = os.path.join(folder, "checkpoint_optimizer."+extension)
    if not os.path.exists(filename):
        print("\tOptimizer parameters not found, skipping initialization")
        return
    print("Loading optimizer parameters from {} ...".format(filename))    
    optimizer.load_state_dict(torch.load(filename))
    if cuda:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
