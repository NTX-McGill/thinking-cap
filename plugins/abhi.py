'''
Based on the csv_collect plugin

Possible arguments: 
train, acquire

Flow : train -> threshold 

acquire -> check threshold value -> event_processor

'''

import csv
import timeit
import datetime
import time
import json
from pprint import pprint
# import numpy as np
import plugin_interface as plugintypes
import requests
import sys, os, signal

class PluginAbhi(plugintypes.IPluginExtended):
    def __init__(self, delim = ",", verbose=False, train=False, acquire=False, person="Typhlosion", url='http://127.0.0.1:5000/', graph=False, recording_session_number=0, window_size=10, attentive=False, duration=1):
        now = datetime.datetime.now()
        self.time_stamp = '%d-%d-%d_%d-%d-%d'%(now.year,now.month,now.day,now.hour,now.minute,now.second)
        self.file_name = self.time_stamp
        self.start_time = timeit.default_timer()
        self.delim = delim
        self.verbose = verbose
        self.train = train
        self.acquire = acquire
        self.training_set = []
        self.person = person
        self.url = url
        self.graph = graph
        self.recording_session_number = recording_session_number
        self.window_size = window_size
        self.last_time = self.start_time
        self.elapsed_time = 0
        self.sample_numbers = 0
        self.attentive = attentive
        self.first_call = True
        self.record_start_time = 0
        self.duration = duration
        self.focus = False
        
    def activate(self):
        if self.graph:
            r = requests.get(self.url+'start') #server
        if len(self.args) > 0:
            if 'no_time' in self.args:
                self.file_name = self.args[0]
            else:
                self.file_name = self.args[0] + '_' + self.file_name;
            if 'verbose' in self.args:
                self.verbose = True
            if 'train' in self.args:
                self.train = True
            if 'acquire' in self.args:
                self.acquire = True
            if 'graph' in self.args:
                self.graph = True
            if 'attentive' in self.args:
                self.attentive = True
            if "focus" in self.args:
                self.focus = True
            if 'duration' in self.args:
                for x in range(0,len(self.args)):
                    if self.args[x]=='duration':
                            try:
                                self.duration = (int)(self.args[x+1])
                            except:
                                self.duration = "Not assigned"
            if 'person' in self.args:
                for x in range(0,len(self.args)):
                    if self.args[x]=='person':
                            try:
                                self.person = self.args[x+1]
                            except:
                                self.person = "Not assigned"
            if 'window_size' in self.args:
                for x in range(0,len(self.args)):
                    if self.args[x]=='window_size':
                            try:
                                self.window_size = ((int)(self.args[x+1]))*1000
                            except:
                                self.window_size = "Not assigned"
            
            if 'recording_session_number' in self.args:
                for x in range(0,len(self.args)):
                    if self.args[x]=='recording_session_number':
                            try:
                                self.recording_session_number = self.args[x+1]
                            except:
                                self.recording_session_number = "Not assigned"

        self.file_name = self.file_name + '.csv'
        # the decimal number is the approximate time between each sample
        pprint("duration is " + str(self.duration))
        self.window_size = (int)(self.duration/0.00014901161193847656)
        # dividing by 30 because sampling rate doesn't match '
        self.window_size = self.window_size/30
        pprint("the window_size is " + str(self.window_size))
        # print "Will export CSV to:", self.file_name
        #Open in append mode
        # with open(self.file_name, 'a') as f:
        # 	f.write('%'+self.time_stamp + '\n')
        
    def deactivate(self):
        print "Done collecting data"
        recording_time = time.time() - self.record_start_time
        if self.graph:
            r = requests.get(self.url+'end') #server
        data = {"data" : self.training_set, "sample_rate" : self.sample_numbers/recording_time}
        att_folder = 'inattentive'
        if self.focus:
            att_folder = self.person
        if self.attentive:
            att_folder = 'attentive'
        with open("data/"+att_folder+"/"+self.person+"_"+str(self.recording_session_number)+'.json','w') as outfile:
            json.dump(data, outfile)
        return

    def show_help(self):
        print "Doesn't store to files as the original module, this is a custom adaptation"

    def acquire_data(self,row):
    # incoming a row of csv data as with timestamp then sample number followed by 8 values for the 8 channels and 3 aux channels data points and one blank value
    # length of the object is 15 right now
        
        res = row.split(",")
        # pprint(len(res))
        timestamp = res[0]
        sample_number = self.sample_numbers
        channel_values = res[3:11]
        aux_values = res[11:14]
        # self.live_graph(channel_values[0])
        res_json = {"timestamp" : timestamp, "sample_number" : sample_number, "channel_values" : channel_values, "aux_values" : aux_values, "delay" : 1}
        self.training_set.append(res_json)
        # pprint(res_json)
        if self.graph:
            r = requests.post(self.url+'data', data=res_json) #server
        # pprint('person name is ' + self.person)


    def threshold(self,values):
    # define a custom threshold function on which to trigger an event
    # should write out a value in json format in a folder that can be read in when actual events need to be triggered
        inp = {}
        filename = self.person
        with open('../datafiles/'+filename+'.json') as infile:
            inp = json.load(infile)
        return ""


    def event_processor(self):
    # add here some API interaction to pause videos or any other things that might be useful
        return ""
    '''
    following doesn't work yet
    '''
    # def live_graph(self,data):
    #     pprint("hello")
    #     ymin = float(min(ydata))-10
    #     ymax = float(max(ydata))+10
    #     plt.ylim([ymin,ymax])
    #     ydata.append(data)
    #     del ydata[0]
    #     line.set_xdata(np.arange(len(ydata)))
    #     line.set_ydata(ydata)  # update the data
    #     plt.draw() # update the plot

    def train(self,row):
    # this function will be used to train and calibrate for the individual 
    # this should return a threshold value that can be used to trigger an event
    # the values obtained from here should be dumped into the class var training set
    # once training is complete, a value should be calculated as to what the threshold is through the threshold function 
    # this value calculation can be done by triggering a separate function that opens the file and does calculation on it
        acquire_data(row)
        return "null"

    def __call__(self, sample):
        if self.first_call:
            self.record_start_time = time.time()
            self.first_call = False
        t = timeit.default_timer() - self.start_time

        '''
        Code below to limit sample collection by time intervals
        '''
        # temp = t - self.last_time
        # self.last_time = t
        # self.elapsed_time += temp
        # pprint(temp)
        # pprint(self.elapsed_time)
        '''
        Code below to limit the number of samples collected 
        '''
        self.sample_numbers += 1
        # pprint(self.sample_numbers)
        if self.sample_numbers > self.window_size:
            pprint(str(self.sample_numbers-1) + " samples have been collected, please type in /stop to finish")
            # sys.exit()
            os.kill(os.getpid(), signal.SIGINT)
            # TODO add code to auto exit the entire plugin 

        #print timeSinceStart|Sample Id
        # if self.verbose:
        #     print("CSV: %f | %d" %(t,sample.id))
        # if self.train:
        #     print ("value of train set to True")
        # if self.acquire:
        #     print ("value of acquire set to True")
        
        # curr = datetime.datetime.now()
        # curr_time = '%d-%d-%d_%d-%d-%d-%d'%(curr.year,curr.month,curr.day,curr.hour,curr.minute,curr.second, curr.microsecond)
        curr_time = time.time()
        row = ''
        row += "Timestamp : " + str(curr_time)
        row += self.delim
        row += str(t)
        row += self.delim
        row += str(sample.id)
        row += self.delim
        for i in sample.channel_data:
            row += str(i)
            row += self.delim
        for i in sample.aux_data:
            row += str(i)
            row += self.delim
        #remove last comma
        self.acquire_data(row)
        row += '\n'
        # with open(self.file_name, 'a') as f:
        # 	f.write(row)



