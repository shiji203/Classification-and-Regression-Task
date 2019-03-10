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

    train_size = len(file1_list)
    train_list = []
    p0_list = {}
    p1_list = {}
    p2_list = {}
    p3_list = {}
    p4_list = {}
    p0_total = 1.0
    p1_total = 1.0
    p2_total = 1.0
    p3_total = 1.0
    p4_total = 1.0
    for i in range(train_size):
        words = file1_list[i].split()
        if int(label_list1[i]) == 1:
            p1_total += len(words)
            for word in words:
                if word not in p1_list.keys():
                    p1_list[word] = 1
                else:
                    p1_list[word] += 1
        elif int(label_list1[i]) == 0:
            p0_total += len(words)
            for word in words:
                if word not in p0_list.keys():
                    p0_list[word] = 1
                else:
                    p0_list[word] += 1
        elif int(label_list1[i]) == 2:
            p2_total += len(words)
            for word in words:
                if word not in p2_list.keys():
                    p2_list[word] = 1
                else:
                    p2_list[word] += 1
        elif int(label_list1[i]) == 3:
            p3_total += len(words)
            for word in words:
                if word not in p3_list.keys():
                    p3_list[word] = 1
                else:
                    p3_list[word] += 1
        else:
            p4_total += len(words)
            for word in words:
                if word not in p4_list.keys():
                    p4_list[word] = 1
                else:
                    p4_list[word] += 1
        train_list.append(words)

    label_list = list(map(int,label_list1))
    label_size = len(label_list)
    # word_list = []
    # for i in range(train_size):
    #     words = train_list[i][:]
    #     word_list.extend(words)

    # word_list = list(set(word_list))
    # word_size = len(word_list)
    #one_hot = []
    # for i in range(train_size):
    #     print("a")
    #     vec = [0] * word_size
    #     words = train_list[i][:]
    #     for word in words:
    #         index = word_list.index(word)
    #         vec[index] = 1
    #     one_hot.append(vec)

    p_0 = 0
    p_1 = 0
    p_2 = 0
    p_3 = 0
    p_4 = 0

    for label in label_list:
        if label == 0:
            p_0 += 1
        elif label == 1:
            p_1 += 1
        elif label == 2:
            p_2 += 1
        elif label == 3:
            p_3 += 1
        elif label == 4:
            p_4 += 1
    p_0 = float(p_0) / label_size
    p_1 = float(p_1) / label_size
    p_2 = float(p_2) / label_size
    p_3 = float(p_3) / label_size
    p_4 = float(p_4) / label_size


    # p1_vec = math.log(p1_list/p1_total)
    # p0_vec = math.log(p0_list/p0_total)
    for key in p1_list.keys():
        p1_list[key] = math.log((float(p1_list[key]) / p1_total)*1000000 )
    for key in p0_list.keys():
        p0_list[key] = math.log((float(p0_list[key]) / p0_total)*1000000 )
    for key in p2_list.keys():
        p2_list[key] = math.log((float(p2_list[key]) / p2_total)*1000000 )
    for key in p3_list.keys():
        p3_list[key] = math.log((float(p3_list[key]) / p3_total)*1000000 )
    for key in p4_list.keys():
        p4_list[key] = math.log((float(p4_list[key]) / p4_total)*1000000 )


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

    correct_num = 0
    for i in range(test_size):
        rate = [0] * 5
        rate[1] = math.log( p_1 )
        rate[0] = math.log( p_0 )
        rate[2] = math.log( p_2 )
        rate[3] = math.log( p_3 )
        rate[4] = math.log( p_4 )

        for word in test_list[i]:
            if word in p1_list.keys():
                rate[1] += p1_list[word]
            if word in p0_list.keys():
                rate[0] += p0_list[word]
            if word in p2_list.keys():
                rate[2] += p2_list[word]
            if word in p3_list.keys():
                rate[3] += p3_list[word]
            if word in p4_list.keys():
                rate[4] += p4_list[word]

        max_index = 0
        max_value = rate[0]
        for k in range(len(rate)):
            if max_value < rate[k]:
                max_value = rate[k]
                max_index = k

        if max_index == int(label_list2[i]):
            correct_num += 1

    print('Accuracy: ', str(float(correct_num) / test_size))

    end = time.clock()
    print("run time: ",str(end-start)," s")

if __name__ == "__main__":
    main()