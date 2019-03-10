import numpy as np
import math
import csv
from pprint import pprint

def main():
    file_name1 = '2/my_train.txt'
    with open(file_name1, 'r', encoding='utf-8') as file1:
        file1_list = file1.readlines()

    train_size = len(file1_list)
    train_list = []
    weights = {}
    for i in range(train_size):
        tmp = file1_list[i].split()
        for tmp1 in tmp:
            weights[tmp1] = 1
        train_list.append(tmp)

    file_name3 = '2/my_train_label.txt'
    with open(file_name3, 'r', encoding='utf-8') as file3:
        label_list = file3.readlines()

    cycles = 15 #迭代次数
    alpha = 1.0 #学习率
    for j in range(cycles):
        correct_num = 0
        print(j+1)
        #随着迭代次数增加，不断减少学习率
        alpha = alpha/(j+1)
        for i in range(train_size):
            tmp_list = train_list[i][:]
            y = 0.0
            #针对句子中的每个词，增加这个词对应的权值
            for k in tmp_list:
                y += (weights[k] * 1)
            #如果y大于等于0，预测为1，y小于0，预测为0
            if y >= 0.0:
                sign_y = 1.0
            else:
                sign_y = 0.0
            #如果预测结果正确，不更新权值
            if sign_y == float(label_list[i]):
                correct_num += 1
                continue
            #预测不正确，更新权值
            else:
                #如果结果为1，则更新值为1.0，否则为-1.0
                if float(label_list[i]) == 1.0:
                    e = 1.0
                else:
                    e = -1.0
                #对每个词进行权值的更新
                for k in tmp_list:
                    weights[k] = weights[k] + e*alpha
        print(float(correct_num) / train_size)

    file_name3 = '2/my_validation_label.txt'
    with open(file_name3, 'r', encoding='utf-8') as file3:
        label_list1 = file3.readlines()

    file_name2 = '2/my_validation.txt'
    with open(file_name2, 'r', encoding='utf-8') as file2:
        file2_list = file2.readlines()

    test_size = len(file2_list)
    test_list = []
    for i in range(test_size):
        tmp = file2_list[i].split()
        test_list.append(tmp)

    accurate_number = 0
    #遍历每一个句子，得到预测结果
    for i in range(test_size):
        tmp_list = test_list[i][:]
        test_y = 0.0
        #对于句子的每一词，如果存在对应的权值，直接相加
        for k in tmp_list:
            if  k in weights.keys():
                test_y += weights[k] * 1
        #如果结果大于0，预测为1，否则预测为0
        if test_y >= 0.0:
            sign_y = 1.0
        else:
            sign_y = 0.0
        if sign_y == float(label_list1[i]):
            accurate_number += 1

    print('Accuracy: ', str(float(accurate_number) / test_size))

if __name__ == "__main__":
    main()