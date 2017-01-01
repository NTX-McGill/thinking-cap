from flask import Flask, request, jsonify, redirect, url_for
from flask_cors import CORS
# from graphing import *
from pprint import pprint
# from bci_workshop_tools import *
from DeepEEG import getState
from subprocess import Popen, PIPE, STDOUT, call, check_call
import json, time, sys
# import user
import os
import requests
import thread, csv

app = Flask(__name__)
person_name = ""
first_recording = True
attentive = 1
time_interval = ""
counter = 0
CORS(app)
@app.route('/')
def hello():
    return "Welcome to the local server"

@app.route('/login', methods=['POST'])
def login():
    temp = request.get_json()
    # TODO: create a file 
    global person_name 
    person_name = temp['name']
    return jsonify({"name" : person_name})

@app.route('/start', methods=['POST'])
def start(sub_sample_duration=0):
    temp = request.get_json()
    attentive_state = temp['attentive']
    # attentive_state will be true / false / focus
    duration_value = "15" # for the demo it is for training
    if attentive_state is "focus":
        duration_value = sub_sample_duration
    # attentive_state will be true / false 
    if attentive_state is "true":
        attentive_state = "attentive"
    elif attentive_state is "false":
        attentive_state = ""

    # fileName = 'data/'
    # if "true" in attentive_state:
    #     fileName = os.join(fileName,'attentive/%s'%person_name)
    # elif "false" in attentive_state:
    #     fileName = os.join(fileName,'inattentive/%s'%person_name)
    # elif "focus" in attentive_state:
    #     fileName = os.join(fileName,'%s/'%person_name)
    

    global person_name
    pprint("person_name is " + person_name)
    #dummy = "-p /dev/tty.usbserial-DB00J8RE --add abhi person " + person_name +attentive_state + " duration " + duration_value
    dummy = "-p /dev/ttyUSB0 --add abhi person " + person_name + " duration " + duration_value + " "+attentive_state

   
    # dummy = "-p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    # args_list = dummy.split(" ")
    # p = Popen(["python", "user.py"] + args_list, stdin=PIPE, stdout=PIPE)
    # time.sleep(20)
    # pid = p.pid
    call(["./start.sh", dummy])
    # out, err = p.communicate(input=b'/start')
    
    global counter 
    pprint("counter is " + str(counter))
    counter +=1

    # p = Popen(["./start.sh"])
    '''
    temp = "python user.py -p /dev/tty.usbserial-DB00J8RE --add abhi person Jake window_size 1 recording_session_number 12 attentive"
    p = Popen(temp.split(" "))
    board = user.giveBoard()
    '''
    # rc = p.poll()
    return "Initializing"

@app.route('/triggerTraining')
def triggerTraining():
    call(["python","DeepEEG.py"])
    return "Done running"

@app.route('/startFocus', methods=['POST'])
def startFocus():
    '''
    loop to call /start
      if elapsed time less than duration time 
      call start , "focus"
      if first_recording
        callEEG
        first_recording =  False 
      '''

    global first_recording
    temp = request.get_json()
    focus_duration = 30#temp['focus_duration'] # 30 seconds for the demo
    sub_sample_duration = 5 
    time_elapsed = 0
    data = {"attentive": "focus"}

    while(time_elapsed<focus_duration):
        time_elapsed += sub_sample_duration
        r = requests.post("http://127.0.0.1:5000/start",json=data)
        if first_recording:
            thread.start_new_thread(callEEG, (sub_sample_duration, focus_duration))
            first_recording = False
    return ""

@app.route('/endFocus')
def endFocus():
    return ""

@app.route('/data', methods=['POST'])
def data():
    x = request.form['sample_number']
    ys = request.form['channel_values']
    delay = request.form['delay']
    li = []
    for y in ys:
        li.append({"x" : x, 'y' : y})
    process(8, li, delay)
    return ""

@app.route('/lineGraphData')
def lineGraphData():
    temp = { "data": [{ "label": "Attentive", "data": [12, 19, 3, 17, 6, 3, 7,1,1,1], "backgroundColor": "rgba(153,255,51,0.4)" }, { "label": "Inattentive", "data": [2, 29, 5, 5, 2, 3, 10,2,2,2], "backgroundColor": "rgba(400,153,0,0.4)" }] }
    return jsonify(temp)

@app.route('/punchCard')
def punchCard():
    with open("./data/"+ person_name+"/"+person_name+".json") as infile: 
        temp = json.load(infile)
    res = temp['result']
    states = []
    for r in res:
        bs = r['brainStates']
        for b in bs:
            states.append(int(b))
    att_count = 0
    inatt_count = 0
    total = 0
    for s in states:
        if (int)(s) == 0:
            inatt_count += 1
        else:
            att_count +=1 
        total += 1
    interval = (int)(total/10)
    vals = []
    for i in range(0,len(states), interval):
        a = 0
        for j in range(i, i+interval):
            a+=states[j]
        vals.append(100 * ((float)(a)/interval))
    #firstRow = ""+
    txtFile = open('../front_runners/EmilySnook.csv','w')
    txtFile.write(',')
    txtFile.close()

    with open('../front_runners/EmilySnook.csv','a') as outfile:
        x = csv.writer(outfile,delimiter=',')
        x.writerow(' 0 1 2 3 4 5 6 7 8 9 10'.split())
        x.writerow(["Today"] + vals)
    return jsonify({"result" : "done"})

@app.route('/end')
def end():

    return "Stop executed"

@app.route('/polling')
def testjson():
    # attentive has a 0 or 1 
    return jsonify({"result" : attentive })

@app.route('/registerPerson', methods=['POST'])
def registerPerson():
    temp = request.get_json()
    global time_interval
    time_interval = temp['time_interval']
    return jsonify({"name" : person_name, "time_interval" : time_interval})

def callEEG(sub_sample_duration, focus_duration):
    # person_full_name, time_interval between samples
    global person_name
    getState(person_name, sub_sample_duration, focus_duration)
    return "done"

@app.route('/readFile')
def test():
    
    history = []
    global person_name
    with open("./data/"+person_name+"/History.txt") as f:
        for line in f:
            lst = line.split("|")
            timestamp = lst[0]
            time_interval = lst[1]
            brainStates = list(lst[2])
            history.append({"timestamp" : timestamp, "time_interval" : time_interval, "brainStates" : brainStates[:-1]})
    with open("./data/"+ person_name+"/"+person_name+".json",'w') as outfile:
        json.dump({"result":history}, outfile)
    return "done"

@app.route('/pieChart')
def pieChart():
    # return percentage of attentive and inattentive
    temp = {}
    with open("./data/EmilySnook/EmilySnook.json") as infile: 
        temp = json.load(infile)
    res = temp['result']
    states = []
    for r in res:
        bs = r['brainStates']
        for b in bs:
            states.append(int(b))
    att_count = 0
    inatt_count = 0
    total = 0
    for s in states:
        if (int)(s) == 0:
            inatt_count += 1
        else:
            att_count +=1 
        total += 1
    return jsonify({"attentive" : (att_count/total)*100, "inattentive" : (inatt_count/total)*100})
    
if __name__ == "__main__":
    app.run(debug=True, threaded=True)