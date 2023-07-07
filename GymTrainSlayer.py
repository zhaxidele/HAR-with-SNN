import sys, os

CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")

from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
torch.set_printoptions(profile="full")
#torch.set_printoptions(profile="default") # reset
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from torch.utils.data.sampler import WeightedRandomSampler

# Define dataset module
class GymDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        # Read inoput and label
        inputIndex = self.samples[index, 0]
        classLabel = self.samples[index, 1]
        # Read input spike
        inputSpikes = snn.io.read2Dspikes(self.path + str(inputIndex.item()) + '.bs2').toSpikeTensor(torch.zeros((2, 5, 7, self.nTimeBins)),samplingTime=self.samplingTime)
        # Create one-hot encoded desired matrix
        desiredClass = torch.zeros((12, 1, 1, 1))
        desiredClass[classLabel, ...] = 1

        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


# Define the network
class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.loihi(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 32, 3, padding=1, weightScale=10)
        #self.conv2 = slayer.conv(32, 128, 3, padding=1, weightScale=50)
        self.conv2 = slayer.conv(32, 64, 3, padding=1, weightScale=10)
        #self.conv3 = slayer.conv(64, 128, 5, padding=2, weightScale=10)
        #self.pool1 = slayer.pool(2, padding = 1)

        self.fc1 = slayer.dense((5 * 7 * 64), 128)
        #self.fc1 = slayer.dense((5 * 7 * 128), 128)
        self.fc2 = slayer.dense(128, 12)
        #self.drop = slayer.dropout(0.4)
        #self.drop = slayer.dropout(0.1)

    def forward(self, spikeInput):


        spike = self.slayer.spikeLoihi(self.conv1(spikeInput))  # 40, 16, 5, 7
        spike = self.slayer.delayShift(spike, 1, Ts=10)
        #print(spike.shape)

        #spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.conv2(spike))  # 40, 32, 5, 7
        spike = self.slayer.delayShift(spike, 1, Ts=10)
        #print(spike.shape)

        #spike = self.slayer.spikeLoihi(self.pool1(spike))  # 4, 3, 32
        #spike = self.slayer.delayShift(spike, 1)
        #print(spike.shape)

        #spike = self.drop(spike)
        #spike = self.slayer.spikeLoihi(self.conv3(spike))  # 7, 1, 128
        #spike = self.slayer.delayShift(spike, 1, Ts=10)

        #spike = self.drop(spike)
        #spike = self.slayer.spikeLoihi(self.conv4(spike))  # 7, 1, 128
        #spike = self.slayer.delayShift(spike, 1, Ts=10)

        #spike = self.drop(spike)
        #spike = self.slayer.spikeLoihi(self.conv3(spike))  # 7, 1, 128
        #spike = self.slayer.delayShift(spike, 1, Ts=10)

        #spike = self.drop(spike)
        #spike = self.slayer.spikeLoihi(self.conv4(spike))  # 7, 1, 512
        #spike = self.slayer.delayShift(spike, 1, Ts=10)

        #spike = self.slayer.spikeLoihi(self.pool3(spike))  # 4, 200, 8
        spike = spike.reshape((spike.shape[0], -1, 1, 1, spike.shape[-1]))          # 40, 1120, 1, 1, 400
        spike = self.slayer.delayShift(spike, 1, Ts=10)
        #print(spike.shape)

        #spike = self.drop(spike)
        spike = self.slayer.spikeLoihi(self.fc1(spike))  # 64
        spike = self.slayer.delayShift(spike, 1, Ts=10)
        #print(spike.shape)

        spike = self.slayer.spikeLoihi(self.fc2(spike))  # 12
        spike = self.slayer.delayShift(spike, 1, Ts=10)
        #print(spike.shape)

        return spike


# Define Loihi parameter generator
def genLoihiParams(net):
    fc1Weights = snn.utils.quantize(net.fc1.weight, 2).squeeze().cpu().data.numpy()
    fc2Weights = snn.utils.quantize(net.fc2.weight, 2).squeeze().cpu().data.numpy()
    conv1Weights = snn.utils.quantize(net.conv1.weight, 2).squeeze().cpu().data.numpy()
    conv2Weights = snn.utils.quantize(net.conv2.weight, 2).squeeze().cpu().data.numpy()
    #conv3Weights = snn.utils.quantize(net.conv3.weight, 2).flatten().cpu().data.numpy()
    #conv4Weights = snn.utils.quantize(net.conv4.weight, 2).flatten().cpu().data.numpy()
    #pool1Weights = snn.utils.quantize(net.pool1.weight, 2).flatten().cpu().data.numpy()
    #pool2Weights = snn.utils.quantize(net.pool2.weight, 2).flatten().cpu().data.numpy()
    #pool3Weights = snn.utils.quantize(net.pool3.weight, 2).flatten().cpu().data.numpy()

    np.save('Trained/fc1.npy', fc1Weights)
    np.save('Trained/fc2.npy', fc2Weights)
    np.save('Trained/conv1.npy', conv1Weights)
    np.save('Trained/conv2.npy', conv2Weights)
    #np.save('Trained/conv3.npy', conv3Weights)
    #np.save('Trained/conv4.npy', conv4Weights)
    #np.save('Trained/pool1.npy', pool1Weights)
    #np.save('Trained/pool2.npy', pool2Weights)
    #np.save('Trained/pool3.npy', pool3Weights)

    #plt.figure(11)
    #plt.hist(fc1Weights, 256)
    #plt.title('fc1 weights')

    #plt.figure(12)
    #plt.hist(fc2Weights, 256)
    #plt.title('fc2 weights')

    #plt.figure(13)
    #plt.hist(conv1Weights, 256)
    #plt.title('conv1 weights')

    #plt.figure(14)
    #plt.hist(conv2Weights, 256)
    #plt.title('conv2 weights')

    #plt.figure(15)
    #plt.hist(conv3Weights, 256)
    #plt.title('conv3 weights')

    #plt.figure(16)
    #plt.hist(conv4Weights, 256)
    #plt.title('conv4 weights')

    #plt.figure(15)
    #plt.hist(pool1Weights, 256)
    #plt.title('pool1 weights')

    #plt.figure(16)
    #plt.hist(pool2Weights, 256)
    #plt.title('pool2 weights')

    #plt.figure(17)
    #plt.hist(pool3Weights, 256)
    #plt.title('pool3 weights')


#######  sample weigth
def generate_sample_weights(training_data, class_weight_dictionary):
    sample_weights = [class_weight_dictionary[np.where(one_hot_row == 1)[0][0]] for one_hot_row in training_data]
    return np.asarray(sample_weights)

if __name__ == '__main__':
    netParams = snn.params('network.yaml')

    # Define the cuda device to run the code on.
    device = torch.device('cuda')
    # deviceIds = [2, 3]

    # Create network instance.
    net = Network(netParams).to(device)
    # net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

    # Create snn loss instance.
    error = snn.loss(netParams, snn.loihi).to(device)

    # Define optimizer module.
    # optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
    optimizer = snn.utils.optim.Nadam(net.parameters(), lr=0.0001, amsgrad=True)

    # Dataset and dataLoader instances.
    trainingSet = GymDataset(datasetPath=netParams['training']['path']['in'],
                                    sampleFile=netParams['training']['path']['train'],
                                    samplingTime=netParams['simulation']['Ts'],
                                    sampleLength=netParams['simulation']['tSample'])
    #trainLoader = DataLoader(dataset=trainingSet, batch_size=40, shuffle=True, num_workers=1)


    ## to create the sampler
    trainLoader_test = DataLoader(dataset=trainingSet, batch_size=trainingSet.__len__(), shuffle=True, num_workers=1)
    # Loading the First Batch and Printing Information
    for idx, batch in enumerate(trainLoader_test):
        #print('Batch index: ', idx)
        #print('Batch size: ', batch[0].size())
        #print('Batch label: ', batch[1].size())
        #print('Batch label: ', batch[1].resize_(batch[1].size()[0], batch[1].size()[1]).size())
        y_all_array = batch[1].detach().cpu().numpy()
        #print(y_all_array.shape)
        #break
    #print(y_all_array.shape[1])
    y_all_array = y_all_array.reshape(y_all_array.shape[0],12)
    #print(y_all_array.shape)
    class_weight_dictionary = {0: 12.0, 1: 12.0, 2: 12.0, 3: 12.0, 4: 12.0, 5: 1.0, 6: 8.0, 7: 36.0, 8: 8.0, 9: 8.0, 10: 8.0, 11: 8.0}
    weight = generate_sample_weights(y_all_array, class_weight_dictionary)
    #print(weight.shape)
    samples_weight = torch.from_numpy(weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))


    #trainLoader = DataLoader(dataset=trainingSet, sampler=sampler, batch_size=40, shuffle=True, num_workers=1)
    trainLoader = DataLoader(dataset=trainingSet, sampler=sampler, batch_size=16, num_workers=1)


    testingSet = GymDataset(datasetPath=netParams['training']['path']['in'],
                                   sampleFile=netParams['training']['path']['test'],
                                   samplingTime=netParams['simulation']['Ts'],
                                   sampleLength=netParams['simulation']['tSample'])
    #testLoader = DataLoader(dataset=testingSet, batch_size=4, shuffle=True, num_workers=1)
    testLoader = DataLoader(dataset=testingSet, batch_size=16, shuffle=True, num_workers=1)

    # Learning stats instance.
    stats = snn.utils.stats()



    '''
    # Visualize the input spikes (first five samples).
    for i in range(2):
        print("i: ", i)
        input, target, label = trainingSet[i]
        #input[input != 0] = 1

        #print("Input: ", input.shape)
        #input = input.reshape(2, 1, 7, 400)
        #input = input.permute(0, 2, 1, 3)
        #list_a = list(set(input[0, :, :, 0].reshape(-1).tolist()))
        #print(list_a)

        spikeEvent = np.argwhere(input.cpu().data.numpy() > 0)
        print("spikeEvent: ", spikeEvent)
        print("spikeEvent: ", spikeEvent.shape)
        tEvent = spikeEvent[:, 0]
        list_a = list(set(tEvent.reshape(-1).tolist()))
        print("P: ", list_a)
        tEvent = spikeEvent[:, 1]
        list_a = list(set(tEvent.reshape(-1).tolist()))
        print("X: ", list_a)
        tEvent = spikeEvent[:, 2]
        list_a = list(set(tEvent.reshape(-1).tolist()))
        print("Y: ", list_a)
        tEvent = spikeEvent[:, 3]
        list_a = list(set(tEvent.reshape(-1).tolist()))
        print("Time: ", list_a)

        #snn.io.showTD(snn.io.spikeArrayToEvent(input.cpu().data.numpy(), samplingTime=10))
        snn.io.showTD(snn.io.spikeArrayToEvent(input.reshape((2, 5, 7, -1)).cpu().data.numpy(), samplingTime=10), preComputeFrames = True, repeat=False)

        #list_a = list(set(input.reshape(2, 1, 7, -1).cpu().data.numpy().reshape(-1).tolist()))
        list_a = list(set(input.reshape(-1).tolist()))
        print(list_a)

        print("Target: ", target.shape)
        print("label: ", label)
    '''

    #'''
    for epoch in range(2):
        # for epoch in range(1):
        tSt = datetime.now()

        # Training loop.
        for i, (input, target, label) in enumerate(trainLoader, 0):
            net.train()

            # Move the input and target to correct GPU.
            input = input.to(device)
            target = target.to(device)

            # Forward pass of the network.
            output = net.forward(input)

            # Gather the training stats.
            stats.training.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.training.numSamples += len(label)

            # Calculate loss.
            loss = error.numSpikes(output, target)

            # Reset gradients to zero.
            optimizer.zero_grad()

            # Backward pass of the network.
            loss.backward()

            # Update weights.
            optimizer.step()

            # Gather training loss stats.
            stats.training.lossSum += loss.cpu().data.item()

            # Display training stats.
            stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

        # Testing loop.
        # Same steps as Training loops except loss backpropagation and weight update.
        for i, (input, target, label) in enumerate(testLoader, 0):
            net.eval()
            with torch.no_grad():
                input = input.to(device)
                target = target.to(device)

            output = net.forward(input)

            stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).data.item()
            stats.testing.numSamples += len(label)

            loss = error.numSpikes(output, target)
            stats.testing.lossSum += loss.cpu().data.item()
            stats.print(epoch, i)

        # Update stats.
        stats.update()
        stats.plot(saveFig=True, path='Trained/')
        if stats.training.bestLoss is True: torch.save(net.state_dict(), 'Trained/GymNet.pt')

    # Save training data
    stats.save('Trained/')
    net.load_state_dict(torch.load('Trained/GymNet.pt'))
    genLoihiParams(net)
    #'''



    '''
    # Plot the results.
    # Learning loss
    plt.figure(1)
    plt.semilogy(stats.training.lossLog, label='Training')
    plt.semilogy(stats.testing.lossLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    #plt.savefig('loss.png')

    # Learning accuracy
    plt.figure(2)
    plt.plot(stats.training.accuracyLog, label='Training')
    plt.plot(stats.testing.accuracyLog, label='Testing')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    #plt.savefig('accuracy.png')

    plt.show()
    '''
