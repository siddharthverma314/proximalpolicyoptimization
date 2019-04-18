import os
import datetime
import termcolor as tc
import pickle


class Logger:
    """Implements a file-based logger"""
    def __init__(self, logdir='../log/'):
        if logdir[-1] != '/':
            logdir += '/'

        name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.logdir = logdir + name
        self.epoch = 0
        self.curr_data = dict()
        os.mkdir(self.logdir)

    def save_config(self):
        #TODO: Implement this later
        pass
    
    def update(self, data):
        self.curr_data.update(data)
        print(self.curr_data)

    def step(self):
        """
        Iterate to the next epoch. Saves all logging data into
        logdir/epoch_num.json.
        """
        with open(f"{self.logdir}/epoch_{self.epoch}.pkl", 'wb') as f:
            pickle.dump(self.curr_data, f)

        self.epoch += 1
        self.curr_data = dict()
        print(tc.colored(f"Epoch {self.epoch}", 'green'))

    def log(self, msg):
        print(msg)
