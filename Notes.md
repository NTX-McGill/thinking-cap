 
* default set of instructions to send to the board on startup via command line
* make a proper package
* requirements.txt for pip
* fix long recording crash (might be hardware or software fix)


## To run the plugin: 
### python user.py -p /dev/tty.usbserial-xxxxx --add abhi person Abhishek 
Make sure that the plugin shows up in the active plugins [abhi]

To start collecting data use /start on the openbci command line 

Proper shutdown sequence is to type in /stop 

By default it doesn't use the plotly plugin, to use it supply True value for graph argument 

## Things to install 
Flask, sklearn, yapsy, plotly, scipy

## Sampling rate values
0.00015592575073242188
0.00015497207641601562
0.00018215179443359375
0.00017786026000976562
0.00017309188842773438
0.0001628398895263672
0.00019121170043945312
0.00016379356384277344
0.00017714500427246094
0.00018906593322753906
0.0001659393310546875
0.00016307830810546875
0.00015783309936523438
0.0001671314239501953
0.00015783309936523438
0.00020003318786621094
0.00017499923706054688
0.00018596649169921875
0.00016617774963378906
0.00015997886657714844
0.0001590251922607422
0.000186920166015625

## How to use the window_size and attentive 
in addition to the instructions in the "To run the plugin" section, if you need to limit the number of samples being collected add the argument "window_size" followed by the number of samples (in thousands) to collect (default value is 10). 
To mark data as attentive pass "attentive" as an argument to the above command, passing nothing means that it will store the data under the inattentive folder under the data folder  
To choose the recording session use recording_session_number followed by a space and the number so it will save to different files 



