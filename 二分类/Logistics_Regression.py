import math
import numpy as np
import csv
import time

def sigmoid(x) :
    try:
        value = 1.0 / (1 + math.exp(-x))
    except OverflowError:
        value = 1
    return value


def Gradient(train_list,label_list,weights):
    train_size = len(train_list)
    alpha = 0.1 #学习率
    id = 1 #迭代的序号
    while True:
        correct_number = 0
        print(id)
        # 不断更新学习率
        alpha = alpha / id
        id += 1
        #遍历所有的句子，不断训练权值
        for j in range(train_size):
            tmp_list = train_list[j][:]
            y = 0.0
            #累加每个词的权值
            for word in tmp_list:
                y += weights[word] * 1
            y = round(y,6)
            #经过sigmoid激活函数处理
            value = sigmoid(y)
            # if (round(value) == int(label_list[j])):
            #     correct_number += 1
            if value > 0.5 and int(label_list[j]) == 1:
                correct_number += 1
                # continue
            if value < 0.5 and int(label_list[j]) == 0:
                correct_number += 1
                # continue
            #误差
            error = float(label_list[j]) - value
            #对句子的每一个词，更新词的权值
            for word in tmp_list:
                weights[word] = weights[word] + alpha * error * 1
        rate = float(correct_number)/train_size
        print( float(correct_number)/train_size )
        if id >= 15:  #迭代次数达到30次，结束训练
            break

    return weights

def main():
    start = time.clock()
    file_name1 = '2/my_train.txt'
    with open(file_name1,'r',encoding='utf-8') as file1:
        file1_list = file1.readlines()

    train_size = len(file1_list)
    train_list = []
    weights = {}
    for i in range(train_size):
        tmp = file1_list[i].split()
        for k in tmp:
            weights[k] = 1
        train_list.append(tmp)

    file_name3 = '2/my_train_label.txt'
    with open(file_name3,'r',encoding='utf-8') as file3:
        label_list = file3.readlines()

    weights = Gradient(train_list,label_list,weights)

    # file_name4 = 'model/2/LR1.txt'
    # file4 = open(file_name4,'w',encoding='utf-8')
    # for i in range(len(weights)):
    #     file4.write(str(weights[i]) + ' ')

    file_name2 = '2/my_validation.txt'
    with open(file_name2, 'r', encoding='utf-8') as file2:
        file2_list = file2.readlines()

    test_size = len(file2_list)
    test_list = []
    for i in range(test_size):
        tmp = file2_list[i].split()
        test_list.append(tmp)

    file_name3 = '2/my_validation_label.txt'
    with open(file_name3, 'r', encoding='utf-8') as file3:
        label_list1 = file3.readlines()

    accurate_number = 0 #统计预测正确的个数
    #遍历测试集
    for i in range(test_size):
        tmp_list = test_list[i][:]
        test_y = 0.0 #初始化预测结果
        #对句子中每个词的权值进行累加
        for word in tmp_list:
            if word in weights.keys():
                test_y += weights[word] * 1
        #对小数点后6位进行四舍五入
        test_y = round(test_y,6)
        #经过sigmoid激活函数后的预测结果与正确结果比较，正确则统计正确数加1
        if round(sigmoid(test_y)) == int(label_list1[i]):
            accurate_number += 1

    print('Accuracy: ', str(float(accurate_number) / test_size))

    end = time.clock()
    print("run time:  ", str(end - start), " s")

if __name__ == "__main__":
    main()