from __future__ import unicode_literals, print_function, division
from scipy.io import wavfile
import numpy as np
from io import open
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

import unicodedata
import string

if __name__ == '__main__':
    def findFiles(path): return glob.glob(path)

    nSamples = np.arange(-32768, 32769)
    nListSamples = nSamples.tolist()
    samplingRate = 44100
    allFiles = []
    sr, correctSound = wavfile.read('Samples/Correct/CorrectAbby.wav')
    Test_sr, Test_Sound = wavfile.read('Samples/Test/TestAbby.wav')
    #Dah_sr, DahSound = wavfile.read('Samples/Test/dah.wav')
    #Hey_sr, HeySound = wavfile.read('Samples/Test/hey.wav')
    #Whoah_sr, WhoahSound = wavfile.read('Samples/Test/whoah.wav')


    for fileName in findFiles('Samples/Train/*.wav'):
        fs, data = wavfile.read(fileName)
        allFiles.append(data) 

    fileSize = len(correctSound)
    inputSize = 2
    outputSize = 2
    hiddenSize = 12
    batch_size = 0
    num_batches = 0
    temp_batchSize = 200

    while (batch_size == 0):
        if((fileSize % temp_batchSize) == 0):
            batch_size = temp_batchSize
            num_batches = int(fileSize/batch_size)
        else:
            temp_batchSize = temp_batchSize + 1

    def ProcessBatches(file_data):
        mat = np.zeros((num_batches, batch_size, inputSize))
        matrix = np.empty_like(mat)
        for i in range(num_batches):
            #print(file_data.shape)
            #print(file_data[(batch_size*i):(batch_size+(batch_size*i))].shape)
            #print(matrix.shape)
            matrix[i] = file_data[(batch_size*i):(batch_size+(batch_size*i))]
        mat_tensor = torch.from_numpy(matrix)
        mat_tensor = mat_tensor.float()
        return Variable(mat_tensor).cuda()


    correctTensor = ProcessBatches(correctSound)
    TestTensor = ProcessBatches(Test_Sound)
    #DahTensor = ProcessBatches(DahSound)
    #HeyTensor = ProcessBatches(HeySound)
    #WhoahTensor = ProcessBatches(WhoahSound)
    #print(correctTensor)

    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()

            self.hidden_size = hidden_size

            self.i2h = nn.Linear(input_size + hidden_size, hidden_size).cuda()
            self.i2o = nn.Linear(input_size + hidden_size, output_size).cuda()
            self.tanh = nn.Tanh().cuda()
            self.relu = nn.ReLU().cuda()

        def forward(self, input, hidden):
            combined = torch.cat((input.cuda(), hidden.cuda()), 1)
            hidden = self.i2h(combined.cuda())
            combined = self.tanh(combined.cuda())
            hidden = self.relu(hidden.cuda())
            output = self.i2o(combined.cuda())
            output = self.tanh(output.cuda())
            return output, hidden

        def initHidden(self):
            hid = torch.randn(batch_size, self.hidden_size)
            #hid = hid.double()
            return Variable(hid).cuda()

    #def SampleToIndex(sample):
    #    #turn into list for find function
    #    return nListSamples.index(sample)

    #def FileToTensor(fileData):
    #    tensor_L = torch.zeros(len(fileData), 1, len(nSamples))
    #    tensor_R = torch.zeros(len(fileData), 1, len(nSamples))
    #    for pos, sample in enumerate(fileData):
    #        tensor_L[pos][0][SampleToIndex(sample[0])] = 1
    #        tensor_R[pos][0][SampleToIndex(sample[1])] = 1
    #    final_tensor = torch.cat([tensor_L, tensor_R], 1)
    #    return final_tensor
    ##now create network
    #rnn = nn.RNN(inputSize, hiddenSize, outputSize, 'tanh', True, True)

    rnn = RNN(inputSize, hiddenSize, outputSize).cuda()

    #CSMat = np.array([correctSound])
    #correctTensor = Variable(torch.from_numpy(CSMat))
    #Test_Sound_Tensor = torch.from_numpy(Test_Sound)
    #newFileName = 'NewSound.wav'

    def initHidden():
        return Variable(torch.randn(2, 9, hiddenSize))

    def randomChoice(l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample():
        file = randomChoice(allFiles)
        fileProc = ProcessBatches(file)
        #file_tensor = Variable(torch.from_numpy(file))
        return fileProc

    criterion = nn.SmoothL1Loss()
    learning_rate = 0.01

    def train(fileTensor):
        out_tensor = torch.zeros(num_batches, batch_size, outputSize)
        out_var = Variable(out_tensor).cuda()
        hidden = rnn.initHidden()
        ####convert float tensor to double tensor or vice versa
        rnn.zero_grad()
        for i in range(num_batches):    
            output, hidden = rnn(fileTensor[i].cuda(), hidden.cuda())
            out_var[i] = output.cuda()
        loss = criterion(out_var.cuda(), correctTensor.cuda())
        loss.backward()
        for p in rnn.parameters():
            p.data.add_(-learning_rate, p.grad.data)

        return out_var, loss.data[0]

    n_iters = 3000
    print_every = 100
    plot_every = 100
    current_loss = 0
    all_losses = []

    for iter in range(1, n_iters + 1):
        file_tensor = randomTrainingExample()
        output, loss = train(file_tensor.cuda())
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            print("Iteration: " + str(iter) + " Error: " + str(loss))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    newFileName = 'RNNUpdated.wav'

    def BatchToFull(batch_tensor):
        matrix = batch_tensor[0]
        for i in range(1, num_batches):
            matrix = torch.cat((matrix, batch_tensor[i]), 0)
        mat_np = matrix.data.numpy()
        return mat_np
    

    def Predict(fileTensor, ExfileName):
        hidden = rnn.initHidden()
        out_tensor = torch.zeros(num_batches, batch_size, outputSize)
        out_var = Variable(out_tensor)
        for i in range(num_batches):
            output, hidden = rnn(fileTensor[i], hidden)
            out_var[i] = output
        out_file = BatchToFull(out_var)
        newFile = wavfile.write(ExfileName, samplingRate, out_file)

    Predict(TestTensor, 'RNNUpdatedAbby3000iters.wav')
    #Predict(DahTensor, 'DahUpdatedV2.wav')
    #Predict(HeyTensor, 'HeyUpdatedV2.wav')
    #Predict(WhoahTensor, 'WhoahUpdatedV2.wav')
    
        

    #fileName = 'guitar.wav'
    #fs, data = wavfile.read(fileName)
    #newName = 'testing.wav'
    #print(type(data))
    #print(type(data[0][0]))
    #print(str(data.shape))
    #print(str(data))

    ##data = data/(2**15)
    #data = data*(2**15)
    #data = data.astype(np.int16)
    #print(type(data[0][0]))
    #print(str(data.shape))
    #print(str(data))
    #file = wavfile.write(newName, fs, data)

