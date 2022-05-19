import numpy as np
import math

def findMax(list):
    max = -2147483647
    index = -1
    for l in range(len(list)):
        if list[l] > max: 
            max = list[l]
            index = l
    return index 

def findMin(list, indices):
    min = 2147483647
    index = -1
    for l in range(len(list)):
        if l in indices:
            if list[l] < min:
                min = list[l]
                index = l
    return index 

def EuclideanDistance(test, data):
    total = 0
    for x in range(data.size):
        total += (test[x] - data[x]) * (test[x] - data[x])
    return total

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    xt = np.loadtxt(Xtrain_file, delimiter=',')
    yt = np.loadtxt(Ytrain_file, skiprows = 0)


    [length, matrixlength] = xt.shape
    tdf = xt#[int(length - length * .1):]
    #tdf = np.loadtxt(test_data_file, delimiter=',')

    [tlength, tmatrixlength] = tdf.shape

    #k = int(math.sqrt(length)/2)
    for k in range(1, tlength):
        prediction = np.zeros(tlength)
        accuracy = tlength

        for t in range(tlength):
            n1 = 0
            d1 = 0
            n2 = 0
            d2 = 0
            n3 = 0
            d3 = 0
            n4 = 0
            d4 = 0
            distances = np.full((k, 2), 2147483647.0)
            maxdistance = -2147483647
            maxindex = -1
            for x in range(length):
                dis = EuclideanDistance(tdf[t], xt[x])
                if(x < k):
                    distances[x][0] = dis
                    distances[x][1] = yt[x]
                    maxdistance = dis if dis > maxdistance else maxdistance
                    maxindex =x 
                    if yt[x] == 1.0:
                        n1 += 1
                        d1 += dis
                    elif yt[x] == 2.0:
                        n2 += 1
                        d2 += dis
                    elif yt[x] == 3.0:
                        n3 += 1
                        d3 += dis
                    elif yt[x] == 4.0:
                        n4 += 1
                        d4 += dis
                    else:
                        print("Traning Data value is not in data set")
                elif(dis < maxdistance):
                    if distances[maxindex][1] == 1.0:
                        n1 -= 1
                        d1 -= maxdistance
                    if distances[maxindex][1] == 2.0:
                        n2 -= 1
                        d2 -= maxdistance
                    if distances[maxindex][1] == 3.0:
                        n3 -= 1
                        d3 -= maxdistance
                    if distances[maxindex][1] == 4.0:
                        n4 -= 1
                        d4 -= maxdistance
                    distances[maxindex][0] = dis
                    distances[maxindex][1] = yt[x]
                    if yt[x] == 1.0:
                        n1 += 1
                        d1 += dis
                    elif yt[x] == 2.0:
                        n2 += 1
                        d2 += dis
                    elif yt[x] == 3.0:
                        n3 += 1
                        d3 += dis
                    elif yt[x] == 4.0:
                        n4 += 1
                        d4 += dis
                    else:
                        print("Unknown Y value was assigned to max")
                    m = findMax([i[0] for i in distances]) 
                    maxdistance = distances[m][0]
                    maxindex = m
            values = [d1, d2, d3, d4]
            frequencies = [n1, n2, n3, n4]
            indices = np.argwhere(frequencies == np.amax(frequencies))
            [s , useless] = indices.shape
            if(s == 1):
                prediction[t] = float(indices[0][0] + 1)
            else:
                prediction[t] = float(findMin(values, indices) + 1)
            if prediction[t] != yt[t]:
                accuracy -= 1

        print(k, accuracy/tlength)
        np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")



if __name__ == "__main__":
    Xtrain_file = "Xtrain.csv"
    Ytrain_file = "Ytrain.csv"
    test_data_file = "test_data.csv"
    pred_file = "pred_file.csv"
    run(Xtrain_file, Ytrain_file, test_data_file, pred_file)