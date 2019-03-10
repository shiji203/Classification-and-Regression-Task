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

    train_array = np.array(train_list)
    train_tag1 = train_array[:, 6]
    train_array = np.delete(train_array, 6, axis=1)
    train_array = np.array(train_array, dtype=float)

    test_array = np.array(test_list)
    test_tag1 = test_array[:, 6]
    test_array = np.delete(test_array, 6, axis=1)
    test_array = np.array(test_array, dtype=float)

    train_size = len(train_array)
    test_size = len(test_array)

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

    train_one_hot = []
    for i in range(train_tag_size):
        tmp = [0] * train_word_size
        for word in train_tag[i]:
            if word in train_word:
                index = train_word.index(word)
                tmp[index] = 1
        train_one_hot.append(tmp)

    test_one_hot = []
    for i in range(test_tag_size):
        tmp = [0] * train_word_size
        for word in test_tag[i]:
            if word in train_word:
                index = train_word.index(word)
                tmp[index] = 1
        test_one_hot.append(tmp)

    for i in range(6):
        word_list = train_array[:,i]
        max_value = max(word_list)
        min_value = min(word_list)
        ran = max_value - min_value
        word_size = len(word_list)
        for k in range(word_size):
            word_list[k] = float(word_list[k] - min_value) / ran
        train_array[:,i] = word_list

    for i in range(6):
        word_list = test_array[:,i]
        max_value = max(word_list)
        min_value = min(word_list)
        ran = max_value - min_value
        word_size = len(word_list)
        for k in range(word_size):
            word_list[k] = float(word_list[k] - min_value) / ran
        test_array[:,i] = word_list

    train_array = np.insert(train_array,0,1,axis=1)
    test_array = np.insert(test_array,0,1,axis=1)

    train_label = train_array[:, -1]
    train_array = np.delete(train_array, -1, axis=1)
    test_label = test_array[:, -1]
    test_array = np.delete(test_array, -1, axis=1)

    negative1 = train_array[:, 3]
    positive1 = train_array[:, 5]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    for i in range(size1):
        total = negative1[i] + positive1[i] + 1
        ne_rate = float(negative1[i]) / total
        po_rate = float(positive1[i]) / total
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)

    train_array = np.insert(train_array, 7, negative1_rate, axis=1)
    train_array = np.insert(train_array, 8, positive1_rate, axis=1)

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
    aver1 = train_array[:, 2]
    size1 = len(negative1)
    negative1_rate = []
    positive1_rate = []
    aver1_rate = []
    for i in range(size1):
        ne_rate = float(negative1[i]) * float(negative1[i])
        po_rate = float(positive1[i]) * float(positive1[i])
        a_rate = math.exp(float(aver1[i]))
        # a_rate = float(aver1[i]) * 0.5
        negative1_rate.append(ne_rate)
        positive1_rate.append(po_rate)
        aver1_rate.append(a_rate)

    train_array = np.insert(train_array, 13, negative1_rate, axis=1)
    train_array = np.insert(train_array, 14, positive1_rate, axis=1)
    train_array = np.insert(train_array, 15, aver1_rate, axis=1)

    negative2 = test_array[:, 3]
    positive2 = test_array[:, 5]
    aver2 = test_array[:, 2]
    size2 = len(negative2)
    negative2_rate = []
    positive2_rate = []
    aver2_rate = []
    for i in range(size2):
        a_rate = math.exp(float(aver2[i]))
        # a_rate =float(aver2[i]) * 0.5
        ne_rate = float(negative2[i]) * float(negative2[i])
        po_rate = float(positive2[i]) * float(positive2[i])
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        aver2_rate.append(a_rate)

    test_array = np.insert(test_array, 13, negative2_rate, axis=1)
    test_array = np.insert(test_array, 14, positive2_rate, axis=1)
    test_array = np.insert(test_array, 15, aver2_rate, axis=1)

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
    avg1 = train_array[:, 2]
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
    avg2 = test_array[:, 2]
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
        ne_rate = float(negative1[i]) / (float(avg1[i]) + float(positive1[i]) + 1)
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
        po_rate = float(positive2[i]) / (float(negative2[i]) + float(avg2[i]) + 1)
        avg_rate = float(avg2[i]) / (float(negative2[i]) + float(positive2[i]) + 1)
        negative2_rate.append(ne_rate)
        positive2_rate.append(po_rate)
        avg2_rate.append(avg_rate)

    test_array = np.insert(test_array, 28, negative2_rate, axis=1)
    test_array = np.insert(test_array, 29, positive2_rate, axis=1)
    test_array = np.insert(test_array, 30, avg2_rate, axis=1)

    train_one_hot = np.array(train_one_hot, dtype=float)
    test_one_hot = np.array(test_one_hot, dtype=float)
    train_array = np.append(train_array, train_one_hot, axis=1)
    test_array = np.append(test_array, test_one_hot, axis=1)

    train_size = len(train_array)
    test_size = len(test_array)

    weight_size = len(train_array[0])
    weights = np.ones(weight_size) #初始化权值
    alpha = 0.05 #学习率
    cycle = 30 #迭代次数
    for i in range(cycle):
        print(i+1)
        alpha = alpha / (i+1) #更新学习率
        total_error = 0
        #遍历所有行
        for k in range(train_size):
            x = train_array[k][:]
            y = np.dot(x , weights) #线性求和
            error = train_label[k] - y #求误差
            weights = weights + alpha * error * x #更新权值
            total_error += error
        avg_error = float(total_error) / train_size
        print(avg_error)

    score_list = []
    test_score_list = []

    print(weights)
    for i in range(test_size):
        x = test_array[i][:]
        y = np.dot(x,weights)
        score_list.append(test_label[i])
        test_score_list.append(y)

    # print(test_score_list)
    s1 = pandas.Series(score_list)
    s2 = pandas.Series(test_score_list)
    corr = s1.corr(s2)
    print("相关系数： ",str(corr))

    end = time.clock()
    print("run time : ",str(end-start)," s")

if __name__  == "__main__":
    main()
