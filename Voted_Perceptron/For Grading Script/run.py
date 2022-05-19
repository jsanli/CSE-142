import numpy as np

def run(Xtrain_file, Ytrain_file, test_data_file, pred_file):
    xt = np.loadtxt(Xtrain_file, delimiter = ',')
    yt = np.loadtxt(Ytrain_file, skiprows = 0)
    print(xt)

    [length, matrixlength] = xt.shape
    tdf = np.loadtxt(test_data_file, delimiter = ',')
    #tdf = xt[int(length - length * .1):]

    w = np.zeros((length, matrixlength), dtype = np.int64)
    c = np.zeros(length)

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
    for x in range(length):
        total = 0
        for kk in range(k):
            total += (c[kk] * np.sign(np.matmul(w[kk], tdf[x])))
        prediction[x] = 1 if np.sign(total) >= 0 else 0

    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

