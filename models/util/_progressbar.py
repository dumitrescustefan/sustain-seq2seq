import time
import re
import sys


class ProgressBar:
    def __init__ (self, total_steps=100, progress_bar_length=40, total_number_of_updates=400):
        # user editable
        self.progress_bar_length = progress_bar_length # char len of progress bar
        self.__ansi_escape = re.compile(r'\x1b[^m]*m')                
        # internals
        self.set(total_steps=total_steps)
        self.__min_number_of_updates = total_number_of_updates        
        self.__nice_step_interval = 1
        if total_steps>self.__min_number_of_updates:            
            self.__nice_step_interval = int(total_steps/self.__min_number_of_updates)
        
    # how many times should we call update in our interval
    def get_nice_step_interval(self):
        return self.__nice_step_interval
    
    # call to reset
    def set(self, total_steps):
        self.__total_steps = total_steps
        self.__start_time = time.time()
        
    def update(self, progress=-1., text=""):
        if progress >= 0.: # check only during iterable usage
            if progress%self.__nice_step_interval!=0:
                return
        
        progress = float(progress) / float(self.__total_steps) if progress > 0 else 1.        
        if not sys.stdout.isatty(): # we are being redirected/piped so no pretty print, write just at progress = 1 (100%)
            if progress>=1.:                                
                sys.stdout.write(self.__ansi_escape.sub('', text)+"\n")
                sys.stdout.flush()    
            return
        # console print         
        if progress >= 1.:        
            bar = "\r{}\033[K".format(text)
        else:
            block = int(round(self.progress_bar_length * progress))
            bar = "\r\033[47m \033[0m\033[43m{}\033[0m{}\033[47m \033[0m {:3.2f}% ETA \033[93m{}\033[0m/\033[93m{}\033[0m {}\033[K".format(" " * block, " " * (self.progress_bar_length - block), round(progress * 100, 2), self._get_elapsed(), self._get_eta(progress), text)
        sys.stdout.write(bar)
        sys.stdout.flush()    
    
    # return Elapsed as string, already pretty_time
    def _get_elapsed(self):
        time_delta = time.time()-self.__start_time        
        return self._pretty_time(seconds = float(time_delta))    
    
    # return ETA as string, already pretty_time
    def _get_eta(self, progress):    
        time_delta = time.time()-self.__start_time        
        return "-" if progress<=0 else self._pretty_time(seconds = float(time_delta)*(1./progress-progress))    

    # pretty print time (like: 3d:12h:10m:2s). input is number of seconds like "time.time()"
    def _pretty_time(self, seconds, granularity=2):
        intervals = (('w', 604800),('d', 86400),('h', 3600),('m', 60),('s', 1))
        result = []    
        seconds = int(seconds)    
        for name, count in intervals:
            value = seconds // count
            if value:
                seconds -= value * count            
                result.append("{}{}".format(value, name))
        return ':'.join(result[:granularity])

