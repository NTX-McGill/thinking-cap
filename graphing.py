import plotly 
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.tools as tls 
import numpy as np
import datetime 
import time
import random
import json

streams = []
traces = []
stream_links = []
end_value = False

def key_fetcher():
    with open('api_keys.json') as infile:
        return json.load(infile)

def initialize(username="", api_key="", stream_ids=[]):
    creds = key_fetcher()
    tls.set_credentials_file(username=creds['username'], api_key=creds['api_key'], stream_ids=creds['tokens'])

def setup_streams(number, maxpoints):
    stream_ids = tls.get_credentials_file()['stream_ids']
    for i in range(0, number):
        stream_id_i = stream_ids[i]
        stream_i = go.Stream(token=stream_id_i,maxpoints=maxpoints)
        streams.append(stream_i)

def setup_traces(number, mode='lines+markers'):
    for i in range(0, number):
        trace_i = go.Scatter(x=[],y=[],mode=mode,stream=streams[i])
        traces.append(trace_i)
    
def setup_plot(title, filename):
    data = go.Data(traces)
    layout = go.Layout(title=title)
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig, filename=filename)

def setup_stream_links(number):
    for i in range(0, number):
        stream_ids = tls.get_credentials_file()['stream_ids']
        s_i = py.Stream(stream_ids[i])
        stream_links.append(s_i)

def process(number, inputs,delay):
    # inputs are as follows [{'x':[], 'y':[]},{'x':[], 'y':[]},{'x':[], 'y':[]}]
    # This method needs to be put in the main loop from where you send data 
    # The following loop plots one point for each of the streams 
    for i in range(0, number):
        s = stream_links[i]
        s.write(dict(x=inputs[i]['x'],y=inputs[i]['y']))
    time.sleep(delay)

# def test():
#     # used to run tests without an actual openbci plugged in
#     initialize()
#     setup_streams(3,80)
#     setup_traces(3)
#     setup_plot('Time-Series','python-streaming')
#     setup_stream_links(3)
#     for i in range(0,3):
#         stream_links[i].open()
#     k = 5
#     i =0 
#     time.sleep(5)
#     q = 0
#     while not end_value: 
#         x = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
#         y = []
#         for p in range(1,4):
#             y_i = q%p
#             y.append(y_i)
#             q = q+1
#         process(3, [{'x':x,'y':y[0]},{'x':x,'y':y[1]},{'x':x,'y':y[2]}],0.5)
#     for i in range(0, 3):
#         stream_links[i].close()

def testrun():
    initialize()
    setup_streams(8,80)
    setup_traces(8)
    setup_plot('Time-Series','python-streaming')
    setup_stream_links(8)
    for i in range(0,8):
        stream_links[i].open()


def stop():
    for i in range(0, 8):
        stream_links[i].close()

# test()


