import time
import math
import operator
import numpy as np
import csv
import pandas

def main ():
    start = time.clock()
    file1_name = 'train.csv'
    with open(file1_name,'r',encoding='utf-8') as file1:
        file1_reader = csv.reader(file1)
        file1_list = list(file1_reader)

    file1_list.pop(0)
    train_list = file1_list[0:60000]
    test_list = file1_list[60000:80000]

    train_array = np.array(train_list) #转化为array
    train_tag1 = train_array[:, 6] #取出第7列Tag
    train_array = np.delete(train_array,6,axis=1) #删除第7列
    train_array = np.array(train_array,dtype=float) #将值由str转为float

    test_array = np.array(test_list) #转化为array
    test_tag1 = test_array[:, 6] #取出第7列Tag
    test_array = np.delete(test_array,6,axis=1) #删除第7列
    test_array = np.array(test_array,dtype=float) #将值由str转为float

    train_tag = []
    test_tag = []

    train_tag_size = len(train_tag1)
    for i in range(train_tag_size):
        word1_list = train_tag1[i].split("'")
        word_length = len(word1_list)
        word2_list = []
        for j in range(1, word_length, 2):
            word2_list.append(word1_list[j])
        train_tag.append(word2_list)

    test_tag_size = len(test_tag1)
    for i in range(test_tag_size):
        word1_list = test_tag1[i].split("'")
        word_length = len(word1_list)
        word2_list = []
        for j in range(1, word_length, 2):
            word2_list.append(word1_list[j])
        test_tag.append(word2_list)

    # train_word1 = []
    # for i in range(train_tag_size):
    #     for word in train_tag[i]:
    #         if word not in train_word1:
    #             train_word1.append(word)
    #
    # train_word = train_word1[200:250]
    #对相关系数有利的词
    train_word = [' Budget Single Room Non Smoking ', ' Business trip ', ' Couple ',
                  ' Deluxe Queen Guestroom ', ' Double Queen Waterfront ',
                  ' Double or Twin Room with Sea View ', ' Duplex ', ' Embassy Suite ',
                  ' Family with young children ', ' Group ', ' King Duplex Suite ',
                  ' King Hilton Sea View ', ' King Room with Balcony ',
                  ' Large Double Room ', ' Leisure trip ', ' Prestige Suite ',
                  ' Privilege Junior Suite with Spa Access ',
                  ' Privilege Room with 1 Queen Size bed ', ' Queen Guestroom ',
                  ' Solo traveler ', ' Stayed 3 nights ', ' Studio King Family ']

    train_word_size = len(train_word)

    train_one_hot = [] #onehot矩阵
    #遍历每个训练集句子
    for i in range(train_tag_size):
        tmp = [0] * train_word_size #初始化全0
        #遍历句子中的每一个词
        for word in train_tag[i]:
            #如果词存在词列表，进行下一步
            if word in train_word:
                index = train_word.index(word) #得出下标
                tmp[index] = 1 #在当前下标上将值置为1
        train_one_hot.append(tmp) #加入到onehot矩阵

    test_one_hot = []
    for i in range(test_tag_size):
        tmp = [0] * train_word_size
        for word in test_tag[i]:
            if word in train_word:
                index = train_word.index(word)
                tmp[index] = 1
        test_one_hot.append(tmp)

    # #标准化
    # for i in range(6):
    #     word_list = train_array[:,i] #每一特征列的数据
    #     max_value = max(word_list) #求该列的最大值
    #     min_value = min(word_list) #求该列的最小值
    #     ran = max_value - min_value #该列的值范围大小
    #     word_size = len(word_list)
    #     for k in range(word_size):
    #         #根据标准化公式，得出每个数据在0-1的比例
    #         word_list[k] = float(word_list[k] - min_value) / ran
    #     train_array[:,i] = word_list #将处理的列更改到train_array
    #
    # for i in range(6):
    #     word_list = test_array[:,i]
    #     max_value = max(word_list)
    #     min_value = min(word_list)
    #     ran = max_value - min_value
    #     word_size = len(word_list)
    #     for k in range(word_size):
    #         word_list[k] = float(word_list[k] - min_value) / ran
    #     test_array[:,i] = word_list

    train_array = np.insert(train_array,0,1,axis=1)
    test_array = np.insert(test_array,0,1,axis=1)

    train_label = train_array[:,-1]
    train_array = np.delete(train_array,-1,axis=1)
    test_label = test_array[:,-1]
    test_array = np.delete(test_array,-1,axis=1)

    #训练集
    negative1 = train_array[:,3] #negative列
    positive1 = train_array[:,5] #positive列
    size1 = len(negative1)
    negative1_rate = [] #存储比率
    positive1_rate = []
    #遍历每一行句子
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1 #求和
        ne_rate = float(negative1[i]) / total #求negative的负比率
        po_rate = float(positive1[i]) / total #求的positive的正比率
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array,7,negative1_rate,axis=1) #将负比率存储在第8列
    train_array = np.insert(train_array,8,positive1_rate,axis=1) #将正比率存储在第9列

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 7, negative2_rate, axis=1)
    test_array = np.insert(test_array, 8, positive2_rate, axis=1)

    negative1 = train_array[:, 5]
    positive1 = train_array[:, 6]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 9, negative1_rate, axis=1)
    train_array = np.insert(train_array, 10, positive1_rate, axis=1)

    negative2 = test_array[:, 5]
    positive2 = test_array[:, 6]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 9, negative2_rate, axis=1)
    test_array = np.insert(test_array, 10, positive2_rate, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 6]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 11, negative1_rate, axis=1)
    train_array = np.insert(train_array, 12, positive1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 6]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 11, negative2_rate, axis=1)
    test_array = np.insert(test_array, 12, positive2_rate, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 5]
    aver1 = train_array[:,2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    aver1_rate = []
    for i in range(size1):
        ne_rate = float(negative1[i]) * float(negative1[i])
        po_rate = float(positive1[i]) * float(positive1[i])
        a_rate = math.exp(float(aver1[i]) )
        # a_rate = float(aver1[i]) * 0.5
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)
        aver1_rate.append(a_rate)

    train_array = np.insert(train_array, 13, negative1_rate, axis=1)
    train_array = np.insert(train_array, 14, positive1_rate, axis=1)
    train_array = np.insert(train_array,15,aver1_rate,axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    aver2 = test_array[:,2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    aver2_rate = []
    for i in range(size2):
        a_rate = math.exp( float(aver2[i]))
        # a_rate =float(aver2[i]) * 0.5
        ne_rate = float(negative2[i]) * float(negative2[i])
        po_rate = float(positive2[i]) * float(positive2[i])
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        aver2_rate.append(a_rate)

    test_array = np.insert(test_array, 13, negative2_rate, axis=1)
    test_array = np.insert(test_array, 14, positive2_rate, axis=1)
    test_array = np.insert(test_array,15,aver2_rate,axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 16, negative1_rate, axis=1)
    train_array = np.insert(train_array, 17, positive1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 16, negative2_rate, axis=1)
    test_array = np.insert(test_array, 17, positive2_rate, axis=1)

    negative1 = train_array[:, 5]
    positive1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 18, negative1_rate, axis=1)
    train_array = np.insert(train_array, 19, positive1_rate, axis=1)

    negative2 = test_array[:, 5]
    positive2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 18, negative2_rate, axis=1)
    test_array = np.insert(test_array, 19, positive2_rate, axis=1)

    negative1 = train_array[:, 6]
    positive1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 20, negative1_rate, axis=1)
    train_array = np.insert(train_array, 21, positive1_rate, axis=1)

    negative2 = test_array[:, 6]
    positive2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)

    test_array = np.insert(test_array, 20, negative2_rate, axis=1)
    test_array = np.insert(test_array, 21, positive2_rate, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 5]
    avg1 = train_array[:,2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    avg1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1 + avg1[i]
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        avg_rate = float(avg1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)
        avg1_rate.append(avg_rate)

    train_array = np.insert(train_array, 22, negative1_rate, axis=1)
    train_array = np.insert(train_array, 23, positive1_rate, axis=1)
    train_array = np.insert(train_array, 24, avg1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    avg2 = test_array[:,2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    avg2_rate = []
    for i in range(size2):
        total = negative2[i] + positive2[i] + 1 + avg2[i]
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        avg_rate = float(avg2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        avg2_rate.append(avg_rate)

    test_array = np.insert(test_array, 22, negative2_rate, axis=1)
    test_array = np.insert(test_array, 23, positive2_rate, axis=1)
    test_array = np.insert(test_array, 24, avg2_rate, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 5]
    avg1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    avg1_rate = []
    for i in range(size1):
        total = negative1[i] * positive1[i] * avg1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        avg_rate = float(avg1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)
        avg1_rate.append(avg_rate)

    train_array = np.insert(train_array, 25, negative1_rate, axis=1)
    train_array = np.insert(train_array, 26, positive1_rate, axis=1)
    train_array = np.insert(train_array, 27, avg1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    avg2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    avg2_rate = []
    for i in range(size2):
        total = negative2[i] * positive2[i] * avg2[i] + 1
        ne_rate = float(negative2[i]) / total
        po_rate = float(positive2[i]) / total
        avg_rate = float(avg2[i]) / total
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        avg2_rate.append(avg_rate)

    test_array = np.insert(test_array, 25, negative2_rate, axis=1)
    test_array = np.insert(test_array, 26, positive2_rate, axis=1)
    test_array = np.insert(test_array, 27, avg2_rate, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 5]
    avg1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    avg1_rate = []
    for i in range(size1):
        ne_rate = float(negative1[i]) / (float(avg1[i])+float(positive1[i]) + 1)
        po_rate = float(positive1[i]) / (float(negative1[i]) + float(avg1[i]) + 1)
        avg_rate = float(avg1[i]) / (float(negative1[i]) + float(positive1[i]) + 1)
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)
        avg1_rate.append(avg_rate)

    train_array = np.insert(train_array, 28, negative1_rate, axis=1)
    train_array = np.insert(train_array, 29, positive1_rate, axis=1)
    train_array = np.insert(train_array, 30, avg1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    avg2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    avg2_rate = []
    for i in range(size2):
        ne_rate = float(negative2[i]) / (float(positive2[i]) + float(avg2[i]) + 1)
        po_rate = float(positive2[i]) / ( float(negative2[i]) + float(avg2[i]) + 1)
        avg_rate = float(avg2[i]) / ( float(negative2[i]) + float(positive2[i]) + 1)
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        avg2_rate.append(avg_rate)

    test_array = np.insert(test_array, 28, negative2_rate, axis=1)
    test_array = np.insert(test_array, 29, positive2_rate, axis=1)
    test_array = np.insert(test_array, 30, avg2_rate, axis=1)

    train_one_hot = np.array(train_one_hot,dtype=float)  #转化float格式
    test_one_hot = np.array(test_one_hot,dtype=float)
    train_array = np.append(train_array,train_one_hot,axis=1) #按照列格式合并两个array
    test_array = np.append(test_array,test_one_hot,axis=1)

    # t1 = train_array**0.5
    # t2 = test_array**0.5
    # train_array = np.append(train_array,t1,axis=1)
    # test_array = np.append(test_array,t2,axis=1)


    # train_array = np.delete(train_array,6,axis=1)
    # test_array = np.delete(test_array,6,axis=1)
    # print(train_array[0])


    # length1 = len(train_array[0]) #列数
    # #遍历所有列数
    # for i in range(length1):
    #     array1 = train_array[:,i]
    #     length2 = len(array1) #列长度
    #     half = int(length2/2) #列中间的下标
    #     arr1 = [] #列表1
    #     arr2 = [] #列表2
    #     #列的上部分处理
    #     for j in range(0,half):
    #         arr1.append(0)
    #         arr2.append(array1[j])
    #     #列的下部分处理
    #     for j in range(half,length2):
    #         arr1.append(array1[j])
    #         arr2.append(0)
    #     #将arr1更新到当前的列位置
    #     train_array[:,i] = np.array(arr1)
    #     #将arr2插入到后面
    #     train_array = np.insert(train_array,i+length1,arr2,axis=1)
    #
    # length1 = len(test_array[0])
    # for i in range(length1):
    #     array1 = test_array[:, i]
    #     length2 = len(array1)
    #     half = int(length2/2)
    #     arr1 = []
    #     arr2 = []
    #     for j in range(0, half):
    #         arr1.append(0)
    #         arr2.append(array1[j])
    #     for j in range(half, length2):
    #         arr1.append(array1[j])
    #         arr2.append(0)
    #     test_array[:, i] = np.array(arr1)
    #     test_array = np.insert(test_array, i + length1, arr2, axis=1)

    train_size = len(train_array)
    test_size = len(test_array)

    train_mat = np.matrix(train_array) #转化矩阵
    train_label_mat = np.matrix(train_label).T #转化为矩阵，求逆
    xtx = train_mat.T * train_mat #求X.T*X
    if np.linalg.det(xtx) != 0.0:
        weights_list = xtx.I*(train_mat.T*train_label_mat) #解析解公式


    score_list = []
    test_score_list = []

    weights = []
    for tmp in range(len(weights_list)):
        weights.append(float(weights_list[tmp][0]))

    print(weights)
    for i in range(test_size):
        x = test_array[i]
        y = np.dot(x,weights)
        score_list.append(test_label[i])
        test_score_list.append(y)

    print(test_score_list)
    s1 = pandas.Series(score_list)
    s2 = pandas.Series(test_score_list)
    corr = s1.corr(s2)
    print("相关系数： ",str(corr))

    end = time.clock()
    print("run time : ",str(end-start)," s")

if __name__  == "__main__":
    main()
