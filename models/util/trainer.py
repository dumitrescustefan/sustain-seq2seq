import sys
sys.path.insert(0, '../..')

import os, subprocess, gc, time
from collections import OrderedDict
import torch
import torch.nn as nn
from models.util.log import Log
from tqdm import tqdm
import numpy as np
from models.util.validation_metrics import evaluate
from models.util.utils import pretty_time
from models.util.lookup import Lookup

def get_freer_gpu():   # TODO: PCI BUS ID not CUDA ID: os.environ['CUDA_VISIBLE_DEVICES']='2'
    try:    
        import numpy as np
        os_string = subprocess.check_output("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True).decode("utf-8").strip().split("\n")    
        memory_available = [int(x.strip().split()[2]) for x in os_string]
        return int(np.argmax(memory_available))
    except:
        print("Warning: Could execute 'nvidia-smi', default GPU selection is id=0")
        return 0

def _plot_attention_weights(X, y, src_i2w, tgt_i2w, attention_weights, epoch, log_object):
    # plot attention weights for the first example of the batch; USE ONLY FOR DEV where len(predicted_y)=len(gold_y)
    # X is a tensor of size [batch_size, x_seq_len] 
    # y is a tensor of size [batch_size, y_seq_len]
    # attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
    
    # X src labels for the first example
    input_labels = []
    X_list = X[0].cpu().tolist()
    for id in X_list:
        input_labels.append(src_i2w[str(id)])
    
    # y tgt labels for the first example    
    y_list = y[0].cpu().tolist()
    output_labels = []
    for id in y_list:
        output_labels.append(tgt_i2w[str(id)])    
    
    # map weights
    data = np.zeros((len(X_list), len(y_list)))
    for i in range(len(attention_weights)): # each timestep i
        attention_over_inputs_at_timestep_i = attention_weights[i][0]
        data[:,i] = attention_over_inputs_at_timestep_i
    
    log_object.plot_heatmap(data, input_labels=input_labels, output_labels=output_labels, epoch=epoch)

def _print_examples(model, loader, seq_len, src_i2w, tgt_i2w):
    (X_sample, X_sample_lenghts, X_sample_mask), (y_sample, y_sample_lenghts, y_sample_mask) = iter(loader).next()
    seq_len = min(seq_len,len(X_sample))
    
    X_sample = X_sample[0:seq_len]
    X_sample_lenghts = X_sample_lenghts[0:seq_len]
    X_sample_mask = X_sample_mask[0:seq_len]
    
    y_sample = y_sample[0:seq_len]
    y_sample_lenghts = y_sample_lenghts[0:seq_len]
    y_sample_mask = y_sample_mask[0:seq_len]
    
    if model.cuda:
        X_sample = X_sample.cuda()
        X_sample_lenghts = X_sample_lenghts.cuda()
        X_sample_mask = X_sample_mask.cuda()
        y_sample = y_sample.cuda()
        y_sample_lenghts = y_sample_lenghts.cuda()
        y_sample_mask = y_sample_mask.cuda()
        
    if hasattr(model.decoder.attention, 'reset_coverage'):
        model.decoder.attention.reset_coverage(X_sample.size()[0], X_sample.size()[1])
    
    model.eval()   
    y_pred_dev_sample, _, _, _ = model.run_batch((X_sample, X_sample_lenghts, X_sample_mask), (y_sample, y_sample_lenghts, y_sample_mask))#model.forward((X_sample, X_sample_lenghts, X_sample_mask), (y_sample, y_sample_lenghts, y_sample_mask))
    y_pred_dev_sample = torch.argmax(y_pred_dev_sample, dim=2)
    
    # print examples
    for i in range(seq_len):        
        print("X   :", end='')
        for j in range(len(X_sample[i])):
            #print(str(X_sample[i][j].item()) + " ", end='')        
            token = str(X_sample[i][j].item())
            
            if token not in src_i2w.keys():
                print(src_i2w['1'] + " ", end='')
            elif token == '3':
                print(src_i2w['3'], end='')   
                break
            else:
                print(src_i2w[token] + " ", end='')
        
        print("\nY   :", end='')
        for j in range(len(y_sample[i])):
            token = str(y_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'] + " ", end='')                
            else:
                print(tgt_i2w[token] + " ", end='')
        print("\nPRED:", end='')
        for j in range(len(y_pred_dev_sample[i])):
            token = str(y_pred_dev_sample[i][j].item())

            if token not in tgt_i2w.keys():
                print(tgt_i2w['1'] + " ", end='')
            elif token == '3':
                print(tgt_i2w['3'] + " ", end='')                
            else:
                print(tgt_i2w[token] + " ", end='')
        print()
        print("-" * 40)


def train(model, train_loader, valid_loader=None, test_loader=None, model_store_path=None,
          resume=False, max_epochs=100000, patience=10, optimizer=None, criterion=None, lr_scheduler=None,
          tf_start_ratio=0., tf_end_ratio=0., tf_epochs_decay=0): # teacher forcing parameters
    if model_store_path is None: # saves model in the same folder as this script
        model_store_path = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)    
    
    log_path = os.path.join(model_store_path,"log")
    log_object = Log(log_path, clear=True)
    log_object.text("Training model: "+model.__class__.__name__)
    log_object.text("\tresume={}, patience={}, teacher_forcing={}->{} in {} epochs".format(resume, patience, tf_start_ratio, tf_end_ratio, tf_epochs_decay), display = False)
    total_params = sum(p.numel() for p in model.parameters())/1000
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)/1000
    log_object.text("\ttotal_parameters={}K, trainable_parameters={}K".format(total_params, trainable_params))
    log_object.text(model.__dict__, display = False)
    log_object.text("", display = False)
        
    print("Working in folder [{}]".format(model_store_path))
    
    if not criterion:   
        criterion = nn.CrossEntropyLoss(ignore_index=model.tgt_lookup.convert_tokens_to_ids(model.tgt_lookup.pad_token))
    
    n_class = len(model.tgt_lookup)
    batch_size = len(train_loader.dataset.X[0])
    current_epoch = 0
    current_patience = patience
    current_epoch_time = "?"
    current_epoch_train_time = "?"
    current_epoch_dev_time = "?"
    current_epoch_test_time = "?"
    best_accuracy = 0.

    # Calculates the decay per epoch. Returns a vector of decays.
    if tf_epochs_decay > 0:
        epoch_decay = np.linspace(tf_start_ratio, tf_end_ratio, tf_epochs_decay)

    if resume: # load checkpoint         
        extra_variables = model.load_checkpoint(model_store_path, extension="last")                
        load_optimizer_checkpoint(optimizer, model.cuda, model_store_path, extension="last")
        if "epoch" in extra_variables:
            current_epoch = extra_variables["epoch"]                        
        log_object.text("Resuming training from epoch {}".format(current_epoch))        
    
    while current_patience > 0 and current_epoch < max_epochs:        
        #mem_report()
        print("_"*120+"\n")             
        
        # teacher forcing ratio for current epoch
        tf_ratio = tf_start_ratio
        if tf_epochs_decay > 0:
            if current_epoch < tf_epochs_decay: 
                tf_ratio = epoch_decay[current_epoch]
            else: 
                tf_ratio = tf_end_ratio        
       
        
        log_object.text("")
        log_object.text("Starting epoch {}: current_patience={}, time_per_epoch={} ({}/{}/{}), tf_ratio={:.4f} ".format(current_epoch, current_patience,  current_epoch_time, current_epoch_train_time, current_epoch_dev_time, current_epoch_test_time, tf_ratio) )
        
        # train
        time_start = time.time()
        model.train()
        total_loss, log_average_loss, total_coverage_loss, log_total_coverage_loss, total_generator_loss, log_total_generator_loss = 0, 0, 0, 0, 0, 0
        t = tqdm(train_loader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [train]", unit="b")
        for batch_index, ((x_batch, x_batch_lenghts, x_batch_mask), (y_batch, y_batch_lenghts, y_batch_mask)) in enumerate(t):        
            #t.set_postfix(loss=log_average_loss, x_len=len(x_batch[0]), y_len=len(y_batch[0]))                        
            if model.cuda:
                x_batch = x_batch.cuda()
                x_batch_lenghts = x_batch_lenghts.cuda()
                x_batch_mask = x_batch_mask.cuda()
                y_batch = y_batch.cuda()
                y_batch_lenghts = y_batch_lenghts.cuda()
                y_batch_mask = y_batch_mask.cuda()
                                    
            optimizer.zero_grad()
            
            output, loss, attention_weights, display_variables = model.run_batch((x_batch, x_batch_lenghts, x_batch_mask), (y_batch, y_batch_lenghts, y_batch_mask), criterion, tf_ratio)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)            
            if lr_scheduler is not None:
                lr_scheduler.step()
            optimizer.step()
             
            total_loss += loss.item()
            log_average_loss = total_loss / (batch_index+1)
            if "coverage_loss" in display_variables:
                total_coverage_loss += display_variables["coverage_loss"]
                log_total_coverage_loss = total_coverage_loss / (batch_index+1)
                total_generator_loss += display_variables["generator_loss"]
                log_total_generator_loss = total_generator_loss / (batch_index+1)
                
            current_scheduler_lr = "-"
            if lr_scheduler is not None:
                current_scheduler_lr = lr_scheduler.get_lr()[0]
            
             # update progress bar
            t_display_dict = OrderedDict()
            if isinstance(display_variables, dict):
                for key in display_variables:
                    t_display_dict[key] = display_variables[key]                                 
            t_display_dict["cur_loss"] = loss.item()
            t_display_dict["loss"] = log_average_loss            
            t_display_dict["x_y_len"] = str(len(x_batch[0]))+"/"+str(len(y_batch[0]))
            t.set_postfix(ordered_dict = t_display_dict)
                                    
            #log_object.var("Loss vs LR (epoch "+str(current_epoch)+")|Loss|LR", batch_index, loss.item(), y_index = 0)
            #log_object.var("Loss vs LR (epoch "+str(current_epoch)+")|Loss|LR", batch_index, current_scheduler_lr, y_index = 1)
            #log_object.draw()
            #log_object.draw(last_quarter=True)
            
            """log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, torch.cuda.memory_allocated()/1024/1024, y_index=0)
            log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, torch.cuda.max_memory_allocated()/1024/1024, y_index=1)
            log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, torch.cuda.memory_cached()/1024/1024, y_index=2)
            log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, torch.cuda.max_memory_cached()/1024/1024, y_index=3)
            log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, torch.cuda.max_memory_cached()/1024/1024, y_index=3)
            log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached|X_len*10", batch_index, len(x_batch[0])*10, y_index=4)
            log_object.draw()
            """            
            del output, x_batch, y_batch, loss #, l2_reg
            #torch.cuda.empty_cache()
            #if model.cuda:                        
            #    torch.cuda.synchronize()
                            
        del t
        gc.collect()
        
        if model.cuda:
            torch.cuda.empty_cache()
        
        #log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached", batch_index+1, torch.cuda.memory_allocated()/1024/1024, y_index=0)
        #log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached", batch_index+1, torch.cuda.max_memory_allocated()/1024/1024, y_index=1)
        #log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached", batch_index+1, torch.cuda.memory_cached()/1024/1024, y_index=2)
        #log_object.var("GPU Memory|Allocated|Max allocated|Cached|Max cached", batch_index+1, torch.cuda.max_memory_cached()/1024/1024, y_index=3)
        #log_object.draw()
        
        log_object.text("\ttraining_loss={}".format(log_average_loss), display = False)
        log_object.var("Loss|Train loss|Validation loss", current_epoch, log_average_loss, y_index=0)        
        log_object.var("Train Loss and Aux Loss|Total loss|Generator loss|Aux loss", current_epoch, log_average_loss, y_index=0)                
        log_object.var("Train Loss and Aux Loss|Total loss|Generator loss|Aux loss", current_epoch, log_total_generator_loss, y_index=1)                
        log_object.var("Train Loss and Aux Loss|Total loss|Generator loss|Aux loss", current_epoch, log_total_coverage_loss, y_index=2)                
        time_train = time.time() - time_start

        # dev        
        time_start = time.time()        
        if valid_loader is not None:
            
            model.eval()
            with torch.no_grad():
                total_loss = 0
                _print_examples(model, valid_loader, batch_size, model.src_lookup.i2w, model.tgt_lookup.i2w)

                t = tqdm(valid_loader, ncols=120, mininterval=0.5, desc="Epoch " + str(current_epoch)+" [valid]", unit="b")
                y_gold = list()
                y_predicted = list()
                
                for batch_index, ((x_batch, x_batch_lenghts, x_batch_mask), (y_batch, y_batch_lenghts, y_batch_mask)) in enumerate(t):                            
                    if model.cuda:
                        x_batch = x_batch.cuda()
                        x_batch_lenghts = x_batch_lenghts.cuda()
                        x_batch_mask = x_batch_mask.cuda()
                        y_batch = y_batch.cuda()
                        y_batch_lenghts = y_batch_lenghts.cuda()
                        y_batch_mask = y_batch_mask.cuda()
                    
                    output, loss, batch_attention_weights, display_variables = model.run_batch((x_batch, x_batch_lenghts, x_batch_mask), (y_batch, y_batch_lenghts, y_batch_mask), criterion, tf_ratio)
            
                    
                    y_predicted_batch = output.argmax(dim=2)
                    y_gold += y_batch.tolist()
                    y_predicted += y_predicted_batch.tolist()                
                    
                    total_loss += loss.data.item()
                    log_average_loss = total_loss / (batch_index+1)
                    
                    # update progress bar
                    t_display_dict = OrderedDict()
                    t_display_dict["loss"] = log_average_loss
                    if isinstance(display_variables, dict):
                        for key in display_variables:
                            t_display_dict[key] = display_variables[key]                     
                    t.set_postfix(ordered_dict = t_display_dict)
                    
                    del output, loss
                    
                    
                if model.cuda:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                        
            log_object.text("\tvalidation_loss={}".format(log_average_loss), display = False)
            log_object.var("Loss|Train loss|Validation loss", current_epoch, log_average_loss, y_index=1)
            
            score = 0.
            if True: #current_epoch%5==0:
                score, eval = evaluate(y_gold, y_predicted, model.tgt_lookup.i2w, cut_at_eos=True, use_accuracy=False, use_bleu=False)            
                #score = 0
                #eval = {}
                #eval["meteor"], eval["rouge_l_f"] = 0, 0
                log_object.var("Average Scores|Dev scores|Test scores", current_epoch, score, y_index=0)            
                #log_object.var("Average Scores|Dev scores|Test scores", current_epoch, 0, y_index=1) # move to test loader
                log_object.var("Sequence Accuracy Scores|Dev scores|Test scores", current_epoch, eval["sar"], y_index=0)            
                
                log_object.text("\tValidation scores: METEOR={:.4f} , ROUGE-L(F)={:.4f}, average={:.4f}, SAR={:.4f}".format(eval["meteor"], eval["rouge_l_f"], score, eval["sar"]))
                score = eval["sar"]
           
            if score > best_accuracy:
                log_object.text("\tBest score = {:.4f}".format(score))
                best_accuracy = score
                model.save_checkpoint(model_store_path, extension="best", extra={"epoch":current_epoch})
                save_optimizer_checkpoint (optimizer, model_store_path, extension="best")            
                current_patience = patience
            
            # plot attention_weights for the first example of the last batch (does not matter which batch)            
            # batch_attention_weights is a list of [batch_size, seq_len] elements, where each element is the softmax distribution for a timestep
            _plot_attention_weights(x_batch, y_batch, model.src_lookup.i2w, model.tgt_lookup.i2w, batch_attention_weights, current_epoch, log_object)
            
            # dev cleanup
            del t, y_predicted_batch, y_gold, y_predicted
            
        else: # disable patience if no dev provided and always save model 
            current_patience = patience
            model.save_checkpoint(model_store_path, "best", extra={"epoch":current_epoch})
            save_optimizer_checkpoint (optimizer, model_store_path, extension="best")
        time_dev = time.time() - time_start
        # end dev
        
        # start test 
        time_test = 0
        # end test
        
        # end of epoch
        log_object.draw()
        #log_object.draw(last_quarter=True) # draw a second graph with last 25% of results
        
        model.save_checkpoint(model_store_path, "last", extra={"epoch":current_epoch})
        save_optimizer_checkpoint (optimizer, model_store_path, extension="last")

        current_epoch += 1
        current_patience -= 1
        current_epoch_time = pretty_time(time_train+time_dev+time_test)
        current_epoch_train_time = pretty_time(time_train)
        current_epoch_dev_time = pretty_time(time_dev)
        current_epoch_test_time = pretty_time(time_test)

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
    
