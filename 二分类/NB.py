import numpy as np
import operator
import time
import math

def main():
    start = time.clock()

    file2_name = 'my_train_label.txt'
    with open(file2_name, 'r', encoding='utf-8') as file2:
        label_list1 = file2.readlines()

    file1_name = 'my_train.txt'
    with open(file1_name,'r',encoding = 'utf-8') as file1:
        file1_list = file1.readlines()

    train_size = len(file1_list) #训练集句子的行数
    train_list = []
    p1_list = {}  #存储label为1的词的频数
    p0_list = {} #存储label为0的词的频数
    p1_total = 1.0 #存储label为1的词总数
    p0_total = 1.0 #存储label为0的词总数
    #遍历每一行句子
    for i in range(train_size):
        words = file1_list[i].split()
        #label为1的处理
        if int(label_list1[i]) == 1:
            #增加词的总数
            p1_total += len(words)
            #增加每个词的频数
            for word in words:
                if word not in p1_list.keys():
                    p1_list[word] = 1
                else:
                    p1_list[word] += 1
        #label为0的处理
        else:
            #增加词的总数
            p0_total += len(words)
            #增加每个词的频数
            for word in words:
                if word not in p0_list.keys():
                    p0_list[word] = 1
                else:
                    p0_list[word] += 1
        train_list.append(words)

    label_list = list(map(int,label_list1))
    label_size = len(label_list)

    p_1 = float(sum(label_list)) / label_size

    #计算每个词在对应label中的频率，加权对数处理
    for key in p1_list.keys():
        p1_list[key] = math.log((float(p1_list[key]) / p1_total)*10000000 )
    for key in p0_list.keys():
        p0_list[key] = math.log((float(p0_list[key]) / p0_total)*10000000 )


    file3_name = 'my_validation.txt'
    with open(file3_name,'r',encoding='utf-8') as file3:
        file3_list = file3.readlines()

    test_size = len(file3_list)
    test_list = []
    for i in range(test_size):
        words = file3_list[i].split()
        test_list.append(words)

    file4_name = 'my_validation_label.txt'
    with open(file4_name,'r',encoding='utf-8') as file4:
        label_list2 = file4.readlines()

    correct_num = 0 #统计准确个数
    for i in range(test_size):
        p1 = math.log( p_1 ) #label为1的可能性
        p0 = math.log( 1.0 - p_1) #label为0的可能性
        #针对每个词，分别累加到对应的可能性
        for word in test_list[i]:
            if word in p1_list.keys():
                p1 += p1_list[word]
            if word in p0_list.keys():
                p0 += p0_list[word]
        #比较可能性大小，可能性大的预测为对应的label
        if p1 > p0:
            test_y = 1
        else :
            test_y = 0
        #预测正确，统计个数加1
        if test_y == int(label_list2[i]):
            correct_num += 1

    print('Accuracy: ', str(float(correct_num) / test_size))

    end = time.clock()
    print("run time: ",str(end-start)," s")

if __name__ == "__main__":
    main()