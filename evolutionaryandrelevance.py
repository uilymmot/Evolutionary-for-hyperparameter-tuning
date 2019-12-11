import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import pandas as pd
import math
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import copy

# Seed our random values
tseed = 19
npseed = 16
torch.manual_seed(tseed)
np.random.seed(npseed)

# data load dataset
data = np.genfromtxt('Data.csv', delimiter=',', dtype=str)

# Strip first two columns
processedData = np.zeros((data.shape[0] - 2, data.shape[1] - 1))
processedData = data[2:data.shape[0]-1,1:]

# np.random.shuffle(processedData)
processedData = processedData.astype(np.float)

# Circular encoding scheme for variables such as roundness
def circularEncode(data):
    newDataS = np.zeros(data.shape)
    newDataC = np.zeros(data.shape)
    sVal = 0
    cVal= np.pi
    for u in np.unique(data):
        s = np.sin(sVal) 
        c = np.cos(sVal) 
        # Care of rounding errors with floats in python
        if ((s < 0.0001 and s > 0) or (s > -0.0001 and s < 0)):
            s = 0
        if ((c < 0.0001 and c > 0) or (c > -0.0001 and c < 0)):
            c = 0
    
        newDataS[data == u] = s
        newDataC[data == u] = c
        sVal += (np.pi / 4)
        cVal += (np.pi / 4)
        
    newDataS = newDataS + 1
    newDataC = newDataC + 1
    newDataS /= 2
    newDataC /= 2
    
    return newDataS, newDataC

# Extract our features from the sparse matrix of original features
grainSize = processedData[:,1:12]
sorting = processedData[:,13:20]
matrix = processedData[:,21:36]
roundness = processedData[:,37:45]
bioturbation = processedData[:,46:49]
laminae = processedData[:,50:59]

# Totalsum is how many values are actually present
totalSum = np.sum(grainSize, axis=1) + np.sum(sorting, axis=1) + np.sum(matrix, axis=1) + np.sum(roundness, axis=1)  + np.sum(bioturbation, axis=1) + np.sum(laminae, axis=1) 

sinencodingRoundness, cosencodingRoundness = circularEncode(np.argmax(roundness, axis=1))

# Put all our features back into a 8x226 matrix
compiledProcessedData = np.zeros((8,226))
compiledProcessedData[0] = processedData[:,0]
compiledProcessedData[1] = np.argmax(grainSize, axis = 1) 
compiledProcessedData[2] = np.argmax(sorting, axis=1) 
compiledProcessedData[3] = np.argmax(matrix, axis=1) 
compiledProcessedData[4] = sinencodingRoundness
compiledProcessedData[5] = cosencodingRoundness
compiledProcessedData[6] = np.argmax(bioturbation, axis=1) 
compiledProcessedData[7] = np.argmax(laminae, axis=1)

# Normalise all values in the range 0-1
for i in range(1, compiledProcessedData.shape[0]):
    if (np.max(compiledProcessedData[i]) > 1):
        compiledProcessedData[i] = compiledProcessedData[i] / np.max(compiledProcessedData[i])

# Create the 4 porosity classes
porosityClasses = np.copy(compiledProcessedData[0])
porosityClasses[compiledProcessedData[0] >= 15] = 3
porosityClasses[(compiledProcessedData[0] >= 10) & (compiledProcessedData[0] < 15)] = 2
porosityClasses[(compiledProcessedData[0] >= 5) & (compiledProcessedData[0] < 10)] = 1
porosityClasses[compiledProcessedData[0] < 5] = 0
compiledProcessedData[0] = porosityClasses

# Drop features with less more than 4 missing features 
cprocessedData = compiledProcessedData[:,totalSum > 3]
compiledProcessedData = cprocessedData.T

# np.random.shuffle(compiledProcessedData)
print(compiledProcessedData[0:3])

# compiledProcessedData = pd.DataFrame(compiledProcessedData)
# print(compiledProcessedData)

np.random.shuffle(compiledProcessedData)

# 109/50 train/test split
train_data = compiledProcessedData[:109]
test_data = compiledProcessedData[109:]

# split training data into input and target
train_input = train_data[:, 1:]
train_target = train_data[:, 0]

# split training data into input and target
# the first 9 columns are features, the last one is target
test_input = test_data[:, 1:]
test_target = test_data[:, 0]

# create Tensors to hold inputs and outputs
X = Variable(torch.Tensor(train_input))
Y = Variable(torch.Tensor(train_target))

X1 = Variable(torch.Tensor(test_input))
Y1 = Variable(torch.Tensor(test_target))

# Computes the F-score of a model given a target valset
def validationAccuracy(tmodel, valsetX, valSetY):
    predTest = np.argmax(tmodel(valsetX).detach().numpy(), axis=1)
    targetTest = valSetY.numpy()
    results = precision_recall_fscore_support(predTest, targetTest)
    return np.mean(results[2])
	
# Computes and outputs precision, recall and f-scores for train and test sets of a model
def testModel(r_model):
	# trainset
	predTain = np.argmax(r_model(X).detach().numpy(), axis=1)
	targetTrain = Y.numpy()
	testConfusion = confusion_matrix(targetTrain, predTain)
	print(testConfusion)
	results = precision_recall_fscore_support(predTain, targetTrain)
	print("Average precision is:", (np.mean(results[0]) * 100), "%")
	print("Average recall is:", (np.mean(results[1]) * 100), "%")
	for i in range(0, 4):
		print("Precision for class:", i, "is: ", results[0][i])
	print("Average f1-score is:", (np.mean(results[2]) * 100), "%")
	print()

	# Test set
	predTest = np.argmax(r_model(X1).detach().numpy(), axis=1)
	targetTest = Y1.numpy()
	testConfusion = confusion_matrix(targetTest, predTest)
	print(testConfusion)
	results = precision_recall_fscore_support(predTest, targetTest)
	print("Average precision is:", (np.mean(results[0]) * 100), "%")
	print("Average recall is:", (np.mean(results[1]) * 100), "%")
	for i in range(0, 4):
		print("Precision for class:", i, "is: ", results[0][i])
	print("Average f1-score is:", (np.mean(results[2]) * 100), "%")

	return np.mean(results[2])
	
# Computes testing score without outputting to terminal
def testModelNoPrint(r_model):
    predTest = np.argmax(r_model(X1).detach().numpy(), axis=1)
    targetTest = Y1.numpy()
    results = precision_recall_fscore_support(predTest, targetTest)
    return np.mean(results[2])

# same as above but with training set
def trainModelNoPrint(r_model):
    predTain = np.argmax(r_model(X).detach().numpy(), axis=1)
    targetTrain = Y.numpy()
    results = precision_recall_fscore_support(predTain, targetTrain)
    return np.mean(results[2])

    
# define our possible activation functions
possibleActivations = [nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.PReLU(), nn.SELU(), nn.Softplus()]

# An example of a representation of our neural network
nnHypers = dict()
nnHypers['lr'] = 0.08
nnHypers['neurons'] = np.array([25, 30])
nnHypers['dropout'] = np.array([0, 0.25])
nnHypers['bias'] = True
nnHypers['epochs'] = 2500
nnHypers['inputsize'] = train_input.shape[1]
nnHypers['activation'] = 6
nnHypers['outputsize'] = 4
nnHypers['lasthidden'] = 30

# Defines a custom neural network depending on a dictionary of hyperparameters given
def constructNetworkFromNNHyperparams(nnHyperParams):
    class HyperNetwork(nn.Module):
        def __init__(self):
            super(HyperNetwork, self).__init__()
            self.hidden = nn.ModuleList()
            self.drops = nn.ModuleList()
            # Adds the connected layers
            for i in range(0, nnHyperParams['neurons'].shape[0]):
                self.hidden.append(torch.nn.Linear(nnHyperParams['inputsize'], nnHyperParams['neurons'][i], nnHyperParams['bias']))
            # Adds the dropout layers
            for i in range(0, nnHyperParams['dropout'].shape[0]):
                self.drops.append(torch.nn.Dropout(nnHyperParams['dropout'][i]))
            
            # ouput layer
            self.out = torch.nn.Linear(nnHyperParams['lasthidden'], nnHyperParams['outputsize'])

        def forward(self, x):
            l1 = x
            # Connect the layers together with dropouts
            for i in range(0, nnHyperParams['neurons'].shape[0]):
                l1 = possibleActivations[nnHyperParams['activation']](self.hidden[i](x)) # apply custom activation function
                if (i < nnHyperParams['dropout'].shape[0]):
                    l1 = self.drops[i](l1)
            l2 = self.out(l1)
            return l2
    return HyperNetwork()

# given representation of a network, train the network on trainset and evaluate it on valset, return this evaluation
def trainAndEvalNetwork(nnHyperParams, trainSetX, trainSetY, valsetX, valSetY):
    modell = constructNetworkFromNNHyperparams(nnHyperParams) # gets the actual network model
    loss_func = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(modell.parameters(), lr=nnHyperParams['lr'])
    all_losses = []
    plt.ion()
    # Standard lab code for training neural network
    for t in range(nnHyperParams['epochs']):
        # clear gradients for next train
        optimiser.zero_grad()
        # pass input x and get prediction
        prediction = modell.forward(trainSetX)
        # calculate loss
        loss = loss_func(prediction, trainSetY.long())
        # perform backward pass
        loss.backward()
        # call the step function on an Optimiser makes an update to its
        # parameters
        optimiser.step()
    modell.eval()
    
    return validationAccuracy(modell, valsetX, valSetY)

# given a representation, return the trained network on trainset 
def trainNetwork(nnHyperParams, trainSetX, trainSetY):
    modell = constructNetworkFromNNHyperparams(nnHyperParams)
    loss_func = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(modell.parameters(), lr=nnHyperParams['lr'])
    all_losses = []
    plt.ion()   
    for t in range(nnHyperParams['epochs']):
        # clear gradients for next train
        optimiser.zero_grad()
        # pass input x and get prediction
        prediction = modell.forward(trainSetX)
        # calculate loss
        loss = loss_func(prediction, trainSetY.long())
        # perform backward pass
        loss.backward()
        # call the step function on an Optimiser makes an update to its
        # parameters
        optimiser.step()
    modell.eval()
    return modell
    
# Our mutation operator, takes a specification and mutates it
def mutateNetwork(nnHyperParameters):
    # we can mutate lr, neurons in a layer, number of layers, dropouts, epochs, activation function, bias training

    newNNHypers = copy.deepcopy(nnHyperParameters)
    # make random no of mutations (one or two)
    noMutations = np.random.randint(1,3) 
    for i in range(0,noMutations):
        itemToMutate = np.random.randint(0, 7)

        upOrDown = np.random.randint(0,2)
    # Change the learning rate of the network
        if (itemToMutate == 0):
            changeLRAmount = np.random.random() * 0.02   
            if (upOrDown == 0):
                newNNHypers['lr'] += changeLRAmount
            else:
                newNNHypers['lr'] -= changeLRAmount

            if (newNNHypers['lr'] <= 0.01):
                newNNHypers['lr'] = 0.01 + changeLRAmount
        # modifies a random dropout layer
        elif (itemToMutate == 1):
            layerToModify = np.random.randint(0, newNNHypers['dropout'].shape[0])

            newNNHypers['dropout'][layerToModify] = np.random.random() * 0.75
                
        # Change the number of neurons within a given layer of the network
        elif (itemToMutate == 2 or itemToMutate == 1):
            layerToMutate = np.random.randint(newNNHypers['neurons'].shape[0])
            currentNoNeurons = newNNHypers['neurons'][layerToMutate]
            neuronChange = int(float(currentNoNeurons / 3) * np.random.random())

            if (upOrDown == 0):
                newNNHypers['neurons'][layerToMutate] += neuronChange
            else:
                newNNHypers['neurons'][layerToMutate] -= neuronChange

            if (newNNHypers['neurons'][layerToMutate] < 7):
                newNNHypers['neurons'][layerToMutate] = 7 + neuronChange
        # add or remove a layer from the network
        elif (itemToMutate == 3):
            newNeuronLayer = np.random.randint(10, 50)

            if (upOrDown == 0 or newNNHypers['neurons'].shape[0] == 1):
                newNNHypers['neurons'] = np.append(newNNHypers['neurons'], newNeuronLayer)
                newNNHypers['lasthidden'] = newNeuronLayer
                newNNHypers['dropout'] = np.append(newNNHypers['dropout'], 0)
            else:
                currentLayers = newNNHypers['neurons']
                removedLayer = np.random.randint(0, currentLayers.shape[0])
                newNNHypers['neurons'] = np.delete(currentLayers, removedLayer)
                newNNHypers['dropout'] = np.delete(newNNHypers['dropout'], removedLayer)
        # increase or decrease the number of epochs the network trains for
        elif (itemToMutate == 4):
            epochsToChange = np.random.randint(250, 500)

            if (upOrDown):
                newNNHypers['epochs'] += epochsToChange
            else:
                newNNHypers['epochs'] -= epochsToChange

            if (newNNHypers['epochs'] < 100):
                newNNHypers['epochs'] = 100 + epochsToChange
            elif (newNNHypers['epochs'] > 3000):
                newNNHypers['epochs'] = 3000 - epochsToChange
        # flip the learning bias switch
        elif (itemToMutate == 5):
            newNNHypers['bias'] = not newNNHypers['bias']
        # change activation function
        elif (itemToMutate == 6):
            randomActivation = np.random.randint(0, len(possibleActivations))
            newNNHypers['activation'] = randomActivation
    
    # Update the last layer statistics so that our network handles it properly
    newNNHypers['lasthidden'] = newNNHypers['neurons'][newNNHypers['neurons'].shape[0] - 1]
    return newNNHypers

# Our crossover operator
def crossOverNetworks(NNHyperP1, NNHyperP2):
    NNHyperOffspring = dict()
    # randomly selects hyperparameters from the parent networks
    NNHyperOffspring['lr'] = NNHyperP1['lr'] if (np.random.randint(0,2) == 0) else NNHyperP2['lr']
    lay = np.random.randint(0,2)
    NNHyperOffspring['neurons'] = NNHyperP1['neurons'] if (lay == 0) else NNHyperP2['neurons']
    NNHyperOffspring['dropout'] = NNHyperP1['dropout'] if (lay == 0) else NNHyperP2['dropout']
    NNHyperOffspring['bias'] = NNHyperP1['bias'] if (np.random.randint(0,2) == 0) else NNHyperP2['bias']
    NNHyperOffspring['epochs'] = int((NNHyperP1['epochs'] + NNHyperP2['epochs']) / 2)
    NNHyperOffspring['inputsize'] = train_input.shape[1]
    NNHyperOffspring['outputsize'] = 4
    NNHyperOffspring['activation'] = NNHyperP1['activation'] if (np.random.randint(0,2) == 0) else NNHyperP2['activation']
    NNHyperOffspring['lasthidden'] = NNHyperOffspring['neurons'][NNHyperOffspring['neurons'].shape[0] - 1]
    
    return NNHyperOffspring
    
# Generates a random representation of a neural network 
def randomHyperNetwork(): 
    NNHy = dict()
    NNHy['lr'] = (np.random.random() * 0.1) + 0.02 # learning rate between 0 and 0.12
    noLayers = np.random.randint(1,3)
    NNHy['neurons'] = list()
    for i in range(0, noLayers):
        NNHy['neurons'].append(np.random.randint(10,50)) # 1 or 2 layers with 10-50 neurons each
    NNHy['neurons'] = np.array(NNHy['neurons'])
    NNHy['dropout'] = list()
    for i in range(0, noLayers):
        NNHy['dropout'].append(np.random.random() * np.random.random()) # random dropout weighted to lower values`
    NNHy['dropout'] = np.array(NNHy['dropout']) 
    NNHy['bias'] = True
    NNHy['epochs'] = np.random.randint(250, 2500) # random epochs between 250 and 2500
    NNHy['inputsize'] = train_input.shape[1]   
    NNHy['outputsize'] = 4
    NNHy['lasthidden'] = NNHy['neurons'][NNHy['neurons'].shape[0] - 1]
    NNHy['activation'] = np.random.randint(0, len(possibleActivations)) # random activation 
    
    return NNHy

torch.manual_seed(tseed)
np.random.seed(npseed)

hallOfFame = list() # stores the best networks
meanScoresAtEachEpoch = list() 
stdScoresAtEachEpoch = list()

def evolutional(number_epochs):
    currentEpochNeuralNets = list()
    for i in range(0,8):
        currentEpochNeuralNets.append(randomHyperNetwork()) # generate the initial population of 8 random networks
    currentBestNetwork = currentEpochNeuralNets[0]
    currentBestNetworkScore = 0
   
    folds = 3 #number of folds for 'cross-validation'
    
    validationSetsX = list()
    validationSetsY = list()
    trainSetsX = list()
    trainSetsY = list()
    
    # create the 3 folds randomly
    for i in range(0,folds):
        np.random.shuffle(train_data)
        np.random.shuffle(train_data)
        # 90:29 split for training and validation
        training, test = train_data[:90,:], train_data[90:,:]
        trainSetsX.append(Variable(torch.Tensor(training[:, 1:])))
        trainSetsY.append(Variable(torch.Tensor(training[:, 0])))
        validationSetsX.append(Variable(torch.Tensor(test[:,1:])))
        validationSetsY.append(Variable(torch.Tensor(test[:,0]))) 
    
    for epoc in range(number_epochs): 
        scores = list()
        for i in range(0, 8):
            currentAverage = 0
            # Train network on each of the folds and evaluate them, taking the average of the evaluations
            for j in range(0,folds):
                currentAverage += trainAndEvalNetwork(currentEpochNeuralNets[i], trainSetsX[j], trainSetsY[j], validationSetsX[j], validationSetsY[j])
            scores.append((currentAverage / folds))
            print(currentEpochNeuralNets[i]," Model, ", i, ' achieved ', scores[i], ' score')
        scores = np.array(scores)
        ind = np.argpartition(scores, -3)[-3:] # take the top three performers
        ind = ind[np.argsort(scores[ind])] # sort them from worst to best of the 3
        print("Winners of epoch ", epoc, ":", ind)
        
        if (scores[ind[2]] > currentBestNetworkScore): # update our bestnetwork
            currentBestNetwork = currentEpochNeuralNets[ind[2]]
            currentBestNetworkScore = scores[ind[2]]
            
        newEpochNeuralNets = list()
        
        nnetone = currentEpochNeuralNets[ind[2]]
        nnettwo = currentEpochNeuralNets[ind[1]]
        nnetthree = currentEpochNeuralNets[ind[0]]

        # Track statistics of this generation
        hallOfFame.append(nnetone)
        meanScoresAtEachEpoch.append(np.mean(scores))
        stdScoresAtEachEpoch.append(np.std(scores))
        
        # the networks that make it through to the next generation
        newEpochNeuralNets.append(nnetone)
        newEpochNeuralNets.append(crossOverNetworks(nnetone, nnettwo)) 
        newEpochNeuralNets.append(crossOverNetworks(nnetone, nnettwo))
        newEpochNeuralNets.append(crossOverNetworks(nnetone, nnetthree))
        newEpochNeuralNets.append(nnettwo)
        newEpochNeuralNets.append(crossOverNetworks(nnettwo, nnetthree))
        newEpochNeuralNets.append(mutateNetwork(nnetthree))
        newEpochNeuralNets.append(randomHyperNetwork())
        
        # randomly mutate the population, except the best model
        pMutate = (number_epochs - epoc) / number_epochs
        print("Probability of mutating is: ", pMutate)
        for i in range(1, 8):
            chance = np.random.random()           
            if (chance < pMutate):
                newEpochNeuralNets[i] = mutateNetwork(newEpochNeuralNets[i])
        
        currentEpochNeuralNets = newEpochNeuralNets

    return currentBestNetwork

bestNet = evolutional(2)

testScores = list()
for i in hallOfFame:
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    testScores.append(testModelNoPrint(trainNetwork(i, X, Y)))

testModel(trainNetwork(bestNet, X, Y))

counter = 0
for i in testScores:
    print("Model ", counter, " has test score of ", i)
    counter+=1

# Drops the least relevant neurons in a two layer network
def dropNeuronBasedOnRelevanceTwoLayerNetwork(cmodelspec, cmodel, cbestacc):
    neuronToDrop = -1
    currentBestAcc = cbestacc
    bestModel = copy.deepcopy(cmodel)
    
	# first layer
    for i in range(0, cmodelspec['neurons'][0]):
        
        r_model = copy.deepcopy(cmodel)

        for p in r_model.children():
            for param in p.parameters():
                param[i] = 0 # zero out this neuron
                break
            break

        newFOneScore = testModelNoPrint(r_model)

		# check if model improves
        if (newFOneScore > currentBestAcc): 
            currentBestAcc = newFOneScore
            neuronToDrop = i
            bestModel = r_model
    
	# second layer
    for i in range(0, cmodelspec['neurons'][1]):
        r_model = copy.deepcopy(cmodel)
        
        counter = 0
        
        for p in r_model.children():
            for param in p.parameters():
                if (counter == 2): # if its the correct layer
                    param[i] = 0 # zero out this neuron
                    break
                counter+=1
            break

        newFOneScore = testModelNoPrint(r_model)

		# check if model improves
        if (newFOneScore > currentBestAcc): 
            currentBestAcc = newFOneScore
            neuronToDrop = i
            bestModel = r_model
        
    print("Dropped neuron: ", neuronToDrop)
    return bestModel

# This code only runs if get a 2 layer network as our final best model, might not be the case on your machine
	
currentBestModel = hallOfFame[np.argmax(testScores)]
bestNetworkTrained = trainNetwork(currentBestModel, X, Y)
testAccra = testModelNoPrint(bestNetworkTrained)	

# Drop some neurons from our model
relevanceModel = dropNeuronBasedOnRelevanceTwoLayerNetwork(currentBestModel, bestNetworkTrained, testAccra)
for i in range(0, 20):
	relevanceModel = dropNeuronBasedOnRelevanceTwoLayerNetwork(currentBestModel, relevanceModel, testModelNoPrint(relevanceModel))

print('relevance dropped model has: ', testModelNoPrint(relevanceModel))
print('original mode has ', testModelNoPrint(bestNetworkTrained))