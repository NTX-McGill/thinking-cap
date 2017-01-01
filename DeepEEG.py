import numpy
import scipy.io as sio
import theano
import lasagne
import theano.tensor as T
import os

from collections import OrderedDict
import pylab #for graphing

import json
from random import shuffle

import time


#TODO:
#have a fully convolutional layer to account for variable recording time

#inputMat = sio.loadmat('%scovertShiftsOfAttention_VPiac.mat'%dataPath)# got this from: http://bnci-horizon-2020.eu/database/data-sets
def getData(dataPath):

	trainingSet =[]

	for patient in os.listdir(dataPath):
		if patient.endswith('.mat'):
			trainingSet.append(patient)

	print ("%i samples found"%len(trainingSet))

	trainOut = [[1,0],[0,1]]*len(trainingSet) #this will contain the actual state of the brain

	data =[]
	for patient in trainingSet:
		temp = sio.loadmat('%s%s'%(dataPath,patient))
		data.append(temp['data']['X'][0][0][:1000])

	data = numpy.stack(data)
	trainOut = numpy.stack(trainOut)
	
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data

#be able to read from an Attentive folder and create their truth values
def getJsonData(dataPath):
	if 'inattentive' in dataPath:
		trainOut = numpy.array([[0,1]]) #this will contain the actual state of the brain: inattentive
	else:
		trainOut = numpy.array([[1,0]]) #this will contain the actual state of the brain: attentive
	data =[]
	res = {}
	with open(dataPath) as infile:
		res = json.load(infile)
	for timeStamp in res['data']:
		data.append(numpy.array(timeStamp['channel_values'],dtype='float32'))	

	data = numpy.stack(data,axis=1)
	data = numpy.resize(data,(data.shape[0],300000))
	trainOut = numpy.tile(trainOut,(data.shape[0],1)) # for the one dimensional convoluions. set 8 to 1 when using multiple dimensional convolutions
	data = OrderedDict(input=numpy.array(data, dtype='float32'), truth=numpy.array(trainOut, dtype='float32'))
	return data


def createModernNetwork(dimensions,input_var):
	#dimensions = (data.shape[0],1,data.shape[1])
	print ("Creating Network...")

	print ('Input Layer:')
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	print '	',lasagne.layers.get_output_shape(network)

	print ('Hidden Layer:')
	network = lasagne.layers.Conv1DLayer(network, num_filters=15, filter_size=(5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool1DLayer(network,pool_size=(2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv1DLayer(network, num_filters=20, filter_size=(5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool1DLayer(network,pool_size=(2))
	print '	',lasagne.layers.get_output_shape(network)

	print ('Output Layer:')
	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	print '	',lasagne.layers.get_output_shape(network)

	return network

def createNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	print ("Creating Network...")
	dense=False
	extraLayerswoutDSample = 0
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	print ('Input Layer:')
	print '	',lasagne.layers.get_output_shape(network)
	print ('Hidden Layer:')

	#extra layers if learning capacity is not reached. e.g the data is high-dimensional
	for i in xrange(extraLayerswoutDSample):
		network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
		print '	',lasagne.layers.get_output_shape(network)

	if dense:
		network = lasagne.layers.DenseLayer(network, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.DropoutLayer(network,p=0.2)
		print '	',lasagne.layers.get_output_shape(network)

		network = lasagne.layers.DenseLayer(network, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.DropoutLayer(network,p=0.2)
		print '	',lasagne.layers.get_output_shape(network)

		network = lasagne.layers.DenseLayer(network, num_units=1000, nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.DropoutLayer(network,p=0.2)
		print '	',lasagne.layers.get_output_shape(network)
	else:
		network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
		print '	',lasagne.layers.get_output_shape(network)

		network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
		print '	',lasagne.layers.get_output_shape(network)
		'''
		network = lasagne.layers.Conv2DLayer(network, num_filters=30, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
		network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
		print '	',lasagne.layers.get_output_shape(network)
		'''
	#extra layers if learning capacity is not reached. e.g the data is high-dimensional
	for i in xrange(extraLayerswoutDSample):
		network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
		print '	',lasagne.layers.get_output_shape(network)	

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	print ('Output Layer:')
	print '	',lasagne.layers.get_output_shape(network)
	#import pudb; pu.db

	return network

def create3LayerNetwork(dimensions, input_var):
	#dimensions = (1,1,data.shape[0],data.shape[1]) #We have to specify the input size because of the dense layer
	#We have to specify the input size because of the dense layer
	print ("Creating Network...")
	network = lasagne.layers.InputLayer(shape=dimensions,input_var=input_var)
	print ('Input Layer:')
	print '	',lasagne.layers.get_output_shape(network)
	print ('Hidden Layer:')
	

	network = lasagne.layers.Conv2DLayer(network, num_filters=15, filter_size=(5,5), pad ='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)

	network = lasagne.layers.Conv2DLayer(network, num_filters=20, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)
	
	network = lasagne.layers.Conv2DLayer(network, num_filters=30, filter_size=(5,5), pad='same',nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network,pool_size=(2, 2))
	print '	',lasagne.layers.get_output_shape(network)
		

	network = lasagne.layers.DenseLayer(network, num_units=2, nonlinearity = lasagne.nonlinearities.softmax)
	print ('Output Layer:')
	print '	',lasagne.layers.get_output_shape(network)
	#import pudb; pu.db

	return network

#---------------------------------For training------------------------------------------
def createTrainer(network,input_var,y):
	print ("Creating Trainer...")
	#output of network
	out = lasagne.layers.get_output(network)
	#get all parameters from network
	params = lasagne.layers.get_all_params(network, trainable=True)
	#calculate a loss function which has to be a scalar
	cost = T.nnet.categorical_crossentropy(out, y).mean()
	#calculate updates using ADAM optimization gradient descent
	updates = lasagne.updates.adam(cost, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
	#theano function to compare brain to their masks with ADAM optimization
	train_function = theano.function([input_var, y], updates=updates) # omitted (, allow_input_downcast=True)
	return train_function

def createValidator(network, input_var, y):
	print ("Creating Validator...")
	#We will use this for validation
	testPrediction = lasagne.layers.get_output(network, deterministic=True)			#create prediction
	testLoss = lasagne.objectives.categorical_crossentropy(testPrediction,y).mean()   #check how much error in prediction
	testAcc = T.mean(T.eq(T.argmax(testPrediction, axis=1), T.argmax(y, axis=1)),dtype=theano.config.floatX)	#check the accuracy of the prediction

	validateFn = theano.function([input_var, y], [testLoss, testAcc])	 #check for error and accuracy percentage
	return validateFn

def saveModel(network,saveLocation='',modelName='brain1'):

	networkName = '%s%s.npz'%(saveLocation,modelName)
	print ('Saving model as %s'%networkName)
	numpy.savez(networkName, *lasagne.layers.get_all_param_values(network))

def loadModel(network, model='brain1.npz'):

	with numpy.load(model) as f:
		param_values = [f['arr_%d' % i] for i in range(len(f.files))] #gets all param values
		lasagne.layers.set_all_param_values(network, param_values)		  #sets all param values
	return network

def validateNetwork(network,input_var,validationSet):
	print ('Validating the network')
	out = lasagne.layers.get_output(network)
	test_fn = theano.function([input_var],out)
	truePos=falsePos=trueNeg=falseNeg = 0
	for sample in validationSet:
		data = getJsonData(sample)
		trainIn = data['input'].reshape([data['input'].shape[0]] + [1] + [data['input'].shape[1]])

		print ("Sample: %s"%sample)
		if test_fn(trainIn)[0,0] == 1:
			print "Prediction: Attentive" 
			if "inattentive" in sample:
				falsePos+=1
			else:
				truePos+=1

		else: 
			print "Prediction: Inattentive"
			if "inattentive" in sample:
				trueNeg+=1
			else:
				falseNeg+=1
	print ('Samples: %s | True positives: %s | False positives: %s | True negatives: %s | False negatives: %s'%(len(validationSet),truePos,falsePos,trueNeg,falseNeg))

#input: person_Name, timeIntervalBetweenSamples
#output: json file with date,personName (YYYY/MM/DD/HH/DD)
def getState(name,timeInterval,recordDuration):
	dataPath = os.path.join('data',name)
	input_var = T.tensor3('input')
	timeDelay = 3 #time to start up bci tool in seconds
	txtFile = open("%s/History.txt"%dataPath,'a')
	txtFile.write(str(time.time()))#time.strftime("%c")
	txtFile.write('|')
	txtFile.write(str(timeInterval))
	txtFile.write('|')
	txtFile.close()
	#print "Loading sample..."%os.path.join(dataPath,'%s.json'%name)
	data = getJsonData(os.path.join(dataPath,'%s.json'%name))

	print ("Creating Network...")
	networkDimensions = ([data['input'].shape[0]] + [1] + [data['input'].shape[1]])
	network  = createModernNetwork(networkDimensions, input_var)
	print ('loading a previously trained model...\n')
	network = loadModel(network,'Emily.npz')
	out = lasagne.layers.get_output(network)
	test_fn = theano.function([input_var],out)
	try:
		timeElapsed = 0
		while(timeElapsed<recordDuration):
			timeElapsed+=timeInterval
			data = getJsonData(os.path.join(dataPath,'%s.json'%name))
			inputSample = data['input'].reshape([data['input'].shape[0]] + [1] + [data['input'].shape[1]])
			prediction = test_fn(inputSample)[0,0]
			txtFile = open("%s/History.txt"%dataPath,'a')
			txtFile.write(str(int(prediction)))
			txtFile.close()
			if prediction == 1:
				print ("Prediction: Attentive") 
			else:
				print ("Prediction: Inattentive")
			
			time.sleep(timeInterval + timeDelay)
		txtFile = open("%s/History.txt"%dataPath,'a')
		txtFile.write('\n')
		txtFile.close()
		print ("Printing to file")
	except KeyboardInterrupt:
		txtFile = open("%s/History.txt"%dataPath,'a')
		txtFile.write('\n')
		txtFile.close()
		print ("Printing to file")

def main():
	dataPath = 'data'
	#dataPath = '/home/rfratila/Desktop/MENTALdata/'
	testReserve = 0.2
	validationReserve = 0.2
	trainingReserve = 1-(testReserve+validationReserve)
	input_var = T.tensor3('input')
	y = T.dmatrix('truth')

	trainFromScratch = True
	epochs = 10
	samplesperEpoch = 10
	trainTime = 0.01 #in hours
	modelName='Emily'
	dataSet = []

	for patient in [dataPath]:

		attentivePath = os.path.join(dataPath,'attentive')
		inattentivePath = os.path.join(dataPath,'inattentive')

		if os.path.exists(attentivePath) and os.path.exists(inattentivePath):
			dataSet += [os.path.join(attentivePath,i) for i in os.listdir(attentivePath)]
			dataSet += [os.path.join(inattentivePath,i) for i in os.listdir(inattentivePath)  if i.endswith('.json')]
			shuffle(dataSet)

	print ("%i samples found"%len(dataSet))
	#This reserves the correct amount of samples for training, testing and validating
	trainingSet = dataSet[:int(trainingReserve*len(dataSet))]
	testSet = dataSet[int(trainingReserve*len(dataSet)):-int(testReserve*len(dataSet))]
	validationSet = dataSet[int(testReserve*len(dataSet) + int(trainingReserve*len(dataSet))):]

	inputDim = getJsonData(trainingSet[0])

	networkDimensions = (inputDim['input'].shape[0],1,inputDim['input'].shape[1])
	network  = createModernNetwork(networkDimensions, input_var)
	trainer = createTrainer(network,input_var,y)

	validator = createValidator(network,input_var,y)

	if not trainFromScratch:
		print ('loading a previously trained model...\n')
		network = loadModel(network,'Emily2Layer300000.npz')


	#print ("Training for %s epochs with %s samples per epoch"%(epochs,samplesperEpoch))
	record = OrderedDict(iteration=[],error=[],accuracy=[])

	print ("Training for %s hour(s) with %s samples per epoch"%(trainTime,samplesperEpoch))
	epoch = 0
	startTime = time.time()
	timeElapsed = time.time() - startTime
	#for epoch in xrange(epochs):            #use for epoch training
	while timeElapsed/3600 < trainTime :     #use for time training
		epochTime = time.time()
		print ("--> Epoch: %d | Time left: %.2f hour(s)"%(epoch,trainTime-timeElapsed/3600))
		for i in xrange(samplesperEpoch):
			chooseRandomly = numpy.random.randint(len(trainingSet))
			data = getJsonData(trainingSet[chooseRandomly])

			trainIn = data['input'].reshape([data['input'].shape[0]] + [1] + [data['input'].shape[1]])

			trainer(trainIn, data['truth'])

		chooseRandomly = numpy.random.randint(len(testSet))
		print ("Gathering data...%s"%testSet[chooseRandomly])
		data = getJsonData(testSet[chooseRandomly])
		#trainIn = data['input'].reshape([1,1] + list(data['input'].shape))
		trainIn = data['input'].reshape([data['input'].shape[0]] + [1] + [data['input'].shape[1]])
		error, accuracy = validator(trainIn, data['truth'])			     #pass modified data through network
		record['error'].append(error)
		record['accuracy'].append(accuracy)
		record['iteration'].append(epoch)
		timeElapsed = time.time() - startTime
		epochTime = time.time() - epochTime
		print ("	error: %s and accuracy: %s in %.2fs\n"%(error,accuracy,epochTime))
		epoch+=1

	validateNetwork(network,input_var,validationSet)

	saveModel(network=network,modelName=modelName)
	#import pudb; pu.db
	#save metrics to pickle file to be opened later and displayed
	import pickle
	data = {'data':record}
	with open('%sstats.json'%modelName,'w') as output:
		#import pudb; pu.db
		pickle.dump(data,output)
	
if __name__ == "__main__":
    main()