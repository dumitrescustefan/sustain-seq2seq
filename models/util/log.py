import matplotlib
matplotlib.use('Agg')
import os, sys, json, time, glob, pprint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime


class Log():
    def __init__ (self, root, experiment=None, clear=False):
        self.clear = clear               
        self.__start_time = time.time()
                
        if experiment==None:
            from time import gmtime, strftime
            experiment=strftime("%Y-%m-%d-%H-%M", gmtime())            
            print("Starting experiment "+experiment)
        
        self.folder = os.path.join(root, experiment)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        self._json_file = os.path.join(self.folder,"var.json")
        self._txt_file = os.path.join(self.folder,"log.txt")
        
        if clear==True:
            # delete json file
            if os.path.exists(self._json_file):
                os.remove(self._json_file) 
            # delete png files
            filelist=glob.glob(os.path.join(self.folder,"*.png"))
            for file in filelist:
                #print(file)
                os.remove(file)
            
            # delete txt files
            filelist=glob.glob(os.path.join(self.folder,"*.txt"))
            for file in filelist:
                #print(file)
                os.remove(file)
    
    def _extract_from_name(self, name): # name = train_loss|Total loss|Cross Entropy Loss|KL Loss
        parts = name.split("|")
        if len(parts)==1:
            return name, [name]
        else:
            return parts[0], parts[1:]        
    
    def var(self, name, x, y, y_index=0):        
        if os.path.exists(self._json_file):
            with open(self._json_file, "r", encoding="utf-8") as f:
                js = json.load(f)
        else:
            js = {}            
        name, legend = self._extract_from_name(name)
        
        # add data
        if name not in js:
            js[name] = []
        js[name].append((x,y,y_index))
        
        # add legend
        name_legend = name+"_legend"
        if name_legend not in js:
            js[name_legend] = []
        js[name_legend] = legend
        
        with open(self._json_file, "w+", encoding="utf-8") as f:
            json.dump(js, f)
    
    def text(self, text=""):
        if isinstance(text, str):
            if text.strip()!="":                
                txt = "[{}] {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), text.rstrip())
            else:
                txt = "\n"            
        elif isinstance(text, dict):
            txt = "[{}] {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), pprint.pformat(text, indent=4, width=200) )
        else:
            txt = "[{}] {}\n".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(text))
        
        with open(self._txt_file, "a", encoding="utf-8") as f:
            f.write(txt)
        
    def draw(self, last_quarter=False):
        try:        
            if not os.path.exists(self._json_file):
                print("Warning, nothing to draw (no json file found!).")
                return False
                
            with open(self._json_file, "r", encoding="utf-8") as f:
                js = json.load(f)
            
            for name in js:
                # skip legend info
                if "_legend" in name:
                    continue 
                
                # determine how many y_indexes are to plot
                max_y_index = 0
                for x,y,y_index in js[name]:
                    if y_index > max_y_index:
                        max_y_index = y_index
                
                # for each dimension
                plt.clf()   
                plt.figure(figsize=(12,10))
                plt.tight_layout()
                plt.title(name + ("" if not last_quarter else " (Last quarter)"))
                
                for current_y_index in range(max_y_index+1):                
                    plt_x = []
                    plt_y = []            
                    for x,y,y_index in js[name]:
                        if y_index == current_y_index:
                            plt_x.append(x)
                            plt_y.append(y)            
                    if last_quarter:
                        l = len(plt_x)
                        plt_x = plt_x[l-int(l/4):]
                        plt_y = plt_y[l-int(l/4):]
                    plt.plot(plt_x, plt_y, alpha=0.6, label=js[name+"_legend"][current_y_index])
                
                ax = plt.gca()
                ax.grid(which='major', axis='both', linestyle='--')
                plt.legend(loc='upper left', frameon=False)
                if last_quarter:
                    filename = name+"-quarter.png"
                else:
                    filename = name+".png"
                plt.savefig(os.path.join(self.folder, filename), bbox_inches='tight')
                plt.close()                
            return True            
        except:
            return False
    

    def plot_heatmaps(self, xs, tensor_name = "", epoch=0):        
        #fig = plt.figure(figsize=(18, 16))             
        data = xs[0,:,:]
        ncols = data.size()[0]
        data = data.detach().cpu().numpy()
        
        fig, axs = plt.subplots(figsize=(22, 20), ncols=ncols)
        fig.subplots_adjust(wspace=0.02)
        for i in range(ncols):
            #ax = axs[i].subplot(ncols,i,1)            
            sns.heatmap(np.transpose(data[i:i+1,:]), cmap="rocket", ax=axs[i], cbar=False, xticklabels=False, yticklabels=False, annot=False)
            
            #fig.colorbar(ax.collections[0], ax=ax,location="left", use_gridspec=False, pad=0.2)            
        plt.title('Tensor '+tensor_name+' at epoch '+str(epoch))
        fig.savefig(os.path.join(self.folder, "tensor_"+tensor_name+"_"+str(epoch).zfill(4)), bbox_inches='tight')
        plt.clf()
        plt.close()
    
    def plot_heatmap(self, x, input_labels=True, output_labels=True, epoch=0): 
        """
            x is a numpy 2d matrix        
            input and output labels are lists of strings, otherwise will be numbers
        """
        fig = plt.figure(figsize=(18, 16))             
        g = sns.heatmap(x, cmap="copper", xticklabels=output_labels, yticklabels=input_labels)
        g.set_yticklabels(g.get_yticklabels(), rotation = 0)
        g.set_xticklabels(g.get_xticklabels(), rotation = 90)
        plt.title('Attention at epoch '+str(epoch))
        plt.tight_layout()            
        fig.savefig(os.path.join(self.folder, "attention_"+str(epoch).zfill(4)), bbox_inches='tight')
        plt.clf()
        plt.close()
        
    # return Elapsed as string, already pretty_time
    def _get_elapsed(self):
        time_delta = time.time()-self.__start_time        
        return self._pretty_time(seconds = float(time_delta))    

    # pretty print time (like: 3d:12h:10m:2s). input is number of seconds like "time.time()"
    def _pretty_time(self, seconds, granularity=3):
        intervals = (('w', 604800),('d', 86400),('h', 3600),('m', 60),('s', 1))
        result = []    
        seconds = int(seconds)    
        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count            
                result.append("{}{}".format(value, name))
        return ':'.join(result[:granularity])

