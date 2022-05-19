import numpy as np
'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def getW (training_data, training_label_data):
    w = np.matmul(training_data.T, training_data)
    w = np.linalg.inv(w)
    x = np.matmul(training_data.T, training_label_data)
    w = np.matmul(w, x)
    return w

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):
    # Reading data
    training_label_data = np.loadtxt(train_label_dir,skiprows=0);
    training_data = np.loadtxt(train_input_dir,skiprows=0)
    test_data = np.loadtxt(test_input_dir,skiprows=0)

    [label_num, ] = np.unique(training_label_data).shape
    [num ,test_num] = (test_data.shape)


    w_arr = np.zeros([label_num, test_num])

    for x in range(label_num):
        training_data_copy = training_data
        training_label_data_copy = training_label_data
        y = 0
        while y < (training_label_data_copy.size):
            if(x == label_num-1):
                if(training_label_data_copy[y] != x and training_label_data_copy[y] != 0):
                    training_data_copy = np.delete(training_data_copy, y, axis=0)
                    training_label_data_copy = np.delete(training_label_data_copy, y)
                    y-=1
            else:
                if(training_label_data_copy[y] != x and training_label_data_copy[y] != x+1):
                    training_data_copy = np.delete(training_data_copy, y, axis = 0)
                    training_label_data_copy = np.delete(training_label_data_copy, y)
                    y-=1
            y+=1
        w_arr[x] = getW(training_data_copy, training_label_data_copy)

    prediction = np.zeros((num, 1), dtype=np.int16)

    for z in range(num):
        iter = 0
        for x in range(label_num - 1):
            if(np.sign(np.matmul(w_arr[x].T, test_data[z])) > 0.0):
                if(iter != label_num-1):
                    iter+=1
            else:
                if(iter == 0):
                    iter = label_num -1
        prediction[z] = (float)(iter)

    



    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    train_input_dir = 'training/training1.txt'
    train_label_dir = 'training/training1_label.txt'
    test_input_dir = "training/testing1.txt"
    pred_file = 'result'
    run(train_input_dir,train_label_dir,test_input_dir,pred_file)
