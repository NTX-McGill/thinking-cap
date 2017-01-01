import json
import numpy
import pylab #for graphing
import pickle
import sys

data = {}
modelName = "Emily2Layer300000stats.pickle"
if len(sys.argv) > 1:
	modelName = sys.argv[1]
with open('%s'%modelName) as infile:
	if '.pickle' in modelName:
		data = pickle.load(infile)
	elif '.json' in modelName:
		data = json.loads(infile)
		data = data['data']
		for key in data.keys():
			data[key] = numpy.array(data[key],dtype='float32')
'''
for key in data.keys():
	data[key] = numpy.array(data[key],dtype='float32')
'''
pylab.plot(data['iteration'],data['error'], '-ro',label='Test Error')
pylab.plot(data['iteration'],data['accuracy'],'-go',label='Test Accuracy')
pylab.xlabel("Iteration")
pylab.ylabel("Error (%)")
pylab.ylim(0,max(data['error']) if max(data['error']) < 20000 else 20000)
pylab.title(modelName)
pylab.legend(loc='upper right')
#pylab.savefig('.png'%modelName)
pylab.show()#enter param False if running in iterative mode