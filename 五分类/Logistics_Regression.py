import math
import numpy as np
import csv
import time
import operator

def sigmoid(x) :
    try:
        value = 1.0 / (1 + math.exp(-x))
    except OverflowError:
        value = 1
    return value


def Gradient(train_list,label_list,weights,label):
    train_size = len(train_list)
    cycles = 200
    alpha = 0.04
    id = 1
    while True:
        correct_number = 0
        print(id)
        alpha = alpha / id
        id += 1
        for j in range(train_size):
            tmp_list = train_list[j][:]
            y = 0.0
            for word in tmp_list:
                y += weights[word] * 1
            y = round(y,6)
            # print(y)
            value = sigmoid(y)
            if int(label_list[j]) == label:
                label_value = 1
            else:
                label_value = 0
            if value > 0.5 and label_value == 1:
                correct_number += 1
                # continue
            if value < 0.5 and label_value == 0:
                correct_number += 1
                # continue
            # if (round(value) == label_value):
            #     correct_number += 1
            error = float(label_value) - value
            for word in tmp_list:
                weights[word] = weights[word] + alpha * error * 1
        rate = float(correct_number)/train_size
        print( float(correct_number)/train_size )
        if id >= 15:
            break

    return weights

def main():
    start = time.clock()
    file_name1 = 'my_train.txt'
    with open(file_name1,'r',encoding='utf-8') as file1:
        file1_list = file1.readlines()

    train_size = len(file1_list)
    train_list = []
    # weights = {}
    for i in range(train_size):
        tmp = file1_list[i].split()
        # for k in tmp:
        #     weights[k] = 1
        train_list.append(tmp)

    file_name3 = 'my_train_label.txt'
    with open(file_name3,'r',encoding='utf-8') as file3:
        label_list = file3.readlines()

    weights_dic = {}
    for k in range(5):
        weights = {}
        for i in range(train_size):
            tmp = train_list[i][:]
            for word in tmp:
                weights[word] = 1
        weights = Gradient(train_list,label_list,weights,k)
        weights_dic[k] = weights

    file_name2 = 'my_validation.txt'
    with open(file_name2, 'r', encoding='utf-8') as file2:
        file2_list = file2.readlines()

    test_size = len(file2_list)
    test_list = []
    for i in range(test_size):
        tmp = file2_list[i].split()
        test_list.append(tmp)

    file_name3 = 'my_validation_label.txt'
    with open(file_name3, 'r', encoding='utf-8') as file3:
        label_list1 = file3.readlines()

    accurate_number = 0
    for i in range(test_size):
        rate_dic = {}
        for k in range(5):
            weights = weights_dic[k]
            tmp_list = test_list[i][:]
            test_y = 0.0
            for word in tmp_list:
                if word in weights.keys():
                    test_y += weights[word] * 1
            test_y = round(test_y,6)
            rate = sigmoid(test_y)
            rate_dic[k] = rate

        sorted_data = sorted(rate_dic.items(), key=operator.itemgetter(1), reverse=True)
        test_label = sorted_data[0][0]
        if int(test_label) == int(label_list1[i]):
            accurate_number += 1

    print('Accuracy: ', str(float(accurate_number) / test_size))

    end = time.clock()
    print("run time:  ", str(end - start), " s")

if __name__ == "__main__":
    main()