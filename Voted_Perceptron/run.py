from operator import matmul
from pydoc import doc
from re import A
import numpy as np

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    xto = np.loadtxt(Xtrain_file, delimiter=',')
    yt = np.loadtxt(Ytrain_file, skiprows = 0)


    [olength, omatrixlength] = xto.shape
    tdf = xto[int(olength - olength * .1):]

    values = [.01, .02, .05, .1, .2, 1.0]
    for val in values:
        xt = xto[:int(val*.9*olength)]
        [length, matrixlength] = xt.shape

        w = np.zeros((olength, matrixlength), dtype = np.int64)
        c = np.zeros(olength)

        k, t = 1, 0 
        while t <= length:   
            for i in range(length):
                if(yt[i] == 1):
                    if (np.matmul(w[k], xt[i])) <= 0:
                        w[k+1] = w[k] + xt[i]
                        c[k+1] = 1
                        k += 1
                    else:
                        c[k] += 1
                else: 
                    if (-1 * np.matmul(w[k], xt[i])) <= 0:
                        w[k+1] = w[k] + (-1 * xt[i])
                        c[k+1] = 1
                        k += 1
                    else:
                        c[k] += 1
                t += 1

        [length, matrixlength] = tdf.shape
        prediction = np.zeros(length)
        accuracy = length
        for x in range(length):
            total = 0
            for kk in range(k):
                total += (c[kk] * np.sign(np.matmul(w[kk], tdf[x])))
            prediction[x] = 1 if np.sign(total) >= 0 else 0
            accuracy = accuracy - 1 if prediction[x] != yt[olength-length+x] else accuracy

        print("Percentage: " + str(val) + " Accuracy: " + str(accuracy/length))

        np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")



if __name__ == "__main__":
    Xtrain_file = "Xtrain.csv"
    Ytrain_file = "Ytrain.csv"
    test_data_file = "test_data.csv"
    pred_file = "pred_file.csv"
    run(Xtrain_file, Ytrain_file, test_data_file, pred_file)