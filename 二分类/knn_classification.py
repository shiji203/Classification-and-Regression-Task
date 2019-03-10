import csv
import math
import time
import operator
import numpy as np
import scipy.spatial.distance as dist
#找众数，得出对应的label
def find_mode(sample_list,sort_distance,weight_coefficient,measure_choice):
    fre = {}
    size = len(sort_distance)

    if measure_choice == 2:
        # 遍历，统计各个label出现的次数
        for i in range(size):
            index = sort_distance[i][0]
            weight = sort_distance[i][1] * 15 + 1 # 赋予距离倒数，距离越小，倒数越大
            emotion = sample_list[index][1]
            if emotion in fre.keys():
                fre[emotion] += math.pow(weight, weight_coefficient)  # 为了让距离更小的有更大的权值，对当前的权值进行幂次运算，增大权重
            else:
                fre[emotion] = math.pow(weight, weight_coefficient)
    else:
        #遍历，统计各个label出现的次数
        for i in range(size):
            index = sort_distance[i][0]
            if sort_distance[i][1] == 0:  #如果距离为0，说明两个句子完全吻合，赋予极大权值
                weight = 10000
            else:
                weight = 1 / (sort_distance[i][1])   #赋予距离倒数，距离越小，倒数越大
            emotion = sample_list[index][1]
            if emotion in fre.keys():
                # 为了让距离更小的有更大的权值，对当前的权值进行幂次运算，增大权重
                fre[emotion] += math.pow(weight,weight_coefficient)
            else:
                fre[emotion] = math.pow(weight,weight_coefficient)

    max_value = max(fre.values())  #找出权重最大的label
    for key,value in fre.items():
        if value == max_value:   #返回权重最大的label
            return key


#建立验证集的onehot矩阵
def build_validation_one_hot(validation_list,word_list):
    validation_one_hot = []
    validation_size = len(validation_list)
    word_size = len(word_list)
    #初始化onehot矩阵
    for i in range(validation_size):
        tmp_list = [0] * word_size
        validation_one_hot.append(tmp_list)
    #构建验证集onehot矩阵
    for i in range(validation_size):
        # validation_list[i] = validation_list[i][1:]
        validation_tmp_list = validation_list[i][0].split()  #用临时列表存储句子
        validation_tmp_list_size = len(validation_tmp_list)
        for j in range(validation_tmp_list_size):
            for k in range(word_size):
                if validation_tmp_list[j] == word_list[k]:
                    validation_one_hot[i][k] = 1
                    break

    return validation_one_hot

def build_one_hot(sample_list,word_list):
    # 建立训练集的onehot矩阵
    size1 = len(sample_list)
    size3 = len(word_list)
    one_hot = []
    for i in range(size1):
        tmp_list = [0] * size3  # 初始化为0
        one_hot.append(tmp_list)
    # 建立onehot矩阵
    for i in range(size1):
        tmp_list = sample_list[i][0].split()
        size2 = len(tmp_list)
        for j in range(size2):
            # for k in range(size3):
            #     if tmp_list[j] == word_list[k]:
            #         one_hot[i][k] = 1
            #         break
            k = word_list.index(tmp_list[j])  # 查找句子中的词语对应在word_list的下标
            one_hot[i][k] = 1
    return one_hot

#求极大距离,但没有意义
def get_great_distance(one_hot,validation_one_hot,x,sample_list_size,word_list_size):
    distance = {}
    for i in range(sample_list_size):
        dis_max = 0
        for j in range(word_list_size):
            if dis_max < abs(one_hot[i][j] - validation_one_hot[x][j]):
                dis_max = abs(one_hot[i][j] - validation_one_hot[x][j])
        distance[i] = dis_max
    return distance

#求曼哈顿距离
def get_manhattan_distance(one_hot,validation_one_hot,x,sample_list_size):
    distance = {}
    for i in range(sample_list_size):
        # dis_tmp = 0
        # for j in range(word_list_size):
        #     dis_tmp += abs(one_hot[i][j] - validation_one_hot[x][j])
        a1 = np.array(one_hot[i])   #将列表转换为numpy的array
        a2 = np.array(validation_one_hot[x])
        distance[i] = np.linalg.norm(a1 - a2, ord=1)   #应用numpy的函数计算
    return distance

#求欧氏距离
def get_euclidean_distance(one_hot,validation_one_hot,x,sample_list_size):

    distance = {}
    for i in range(sample_list_size):
        # dis_tmp = 0
        # for j in range(word_list_size):
        #     dis_tmp += math.pow((one_hot[i][j] - validation_one_hot[x][j]), 2)
        # dis_tmp = math.pow(dis_tmp, 0.5)
        a1 = np.array(one_hot[i])    #将列表转换为numpy的array
        a2 = np.array(validation_one_hot[x])
        dis_tmp = np.linalg.norm(a1 - a2)  #应用numpy函数计算二范数即欧式距离
        distance[i] = dis_tmp
    return distance

#求汉明距离 两个等长字符串s1与s2之间的汉明距离定义为将其中一个变为另外一个所需要作的最小替换次数
def get_hanming_distance(one_hot,validation_one_hot,x,sample_list_size):
    distance = {}
    for i in range(sample_list_size):
        a1 = np.array(one_hot[i])  #转换为array
        a2 = np.array(validation_one_hot[x])
        smstr = np.nonzero(a1-a2)
        distance[i] = np.shape(smstr[0])[0]
    return distance

#求杰卡德距离，杰卡德距离用两个集合中不同元素占所有元素的比例来衡量两个集合的区分度。
def get_jake_distance(one_hot,validation_one_hot,x,sample_list_size):
    distance = {}
    for i in range(sample_list_size):
        a1 = np.array(one_hot[i])   #转换为array
        a2 = np.array(validation_one_hot[x])
        matv = np.array([a1,a2])
        distance[i] = dist.pdist(matv,'jaccard')
    return distance

#余弦相似度
def get_cosine_similarity(one_hot,validation_one_hot,x,sample_list_size):
    cosine_similarity = {}
    for i in range(sample_list_size):
        # vector_product = 0
        # mode_tmp1 = 0
        # mode_tmp2 =0
        # for j in range(word_list_size):
        #     vector_product += one_hot[i][j] * validation_one_hot[x][j]
        #     mode_tmp1 += one_hot[i][j] * one_hot[i][j]
        #     mode_tmp2 += validation_one_hot[x][j] * validation_one_hot[x][j]
        # mode_tmp1 = math.pow(mode_tmp1,0.5)
        # mode_tmp2 = math.pow(mode_tmp2,0.5)
        # mode = mode_tmp1 * mode_tmp2
        # if mode == 0:
        #     cosine_similarity[i] = 0
        # else:
        #     cosine_similarity[i] = vector_product / (mode_tmp1 * mode_tmp2)
        a1 = np.array(one_hot[i])  # 转换为array
        a2 = np.array(validation_one_hot[x])
        m = np.linalg.norm(a1) * (np.linalg.norm(a2))
        if (m != 0):
            cosine_similarity[i] = np.dot(a1, a2) / m  # 应用余弦公式
        else:
            cosine_similarity[i] = 0
    return cosine_similarity

def measure(one_hot,validation_one_hot,x,size1,measure_choice):
    if measure_choice == 0:
        distance = get_manhattan_distance(one_hot,validation_one_hot,x,size1)  #计算曼哈顿距离
        sorted_data = sorted(distance.items(), key=operator.itemgetter(1))  # 距离排序
    elif measure_choice == 1:
        distance = get_euclidean_distance(one_hot,validation_one_hot,x,size1)  #计算欧式距离
        sorted_data = sorted(distance.items(), key=operator.itemgetter(1))  # 距离排序
    elif measure_choice == 2:
        cosine_similarity = get_cosine_similarity(one_hot,validation_one_hot,x,size1)  #计算余弦相似度
        sorted_data = sorted(cosine_similarity.items(), key=operator.itemgetter(1),reverse=True)  #余弦相似度排序
    elif measure_choice == 3:
        distance = get_hanming_distance(one_hot,validation_one_hot,x,size1)  #计算汉明距离
        sorted_data = sorted(distance.items(), key=operator.itemgetter(1))  # 距离排序
    else:
        distance = get_jake_distance(one_hot, validation_one_hot, x, size1)  # 杰卡德距离
        sorted_data = sorted(distance.items(), key=operator.itemgetter(1))  # 距离排序

    return sorted_data

def get_distance(sample_list,validation_list,index):
    distance = {}
    sample_size = len(sample_list)
    validation_list_index_size = len(validation_list[index])
    for i in range(sample_size):
        tmp_distance = len(sample_list[i]) + validation_list_index_size
        count = 0
        for k in range(validation_list_index_size):
            if validation_list[index][k] in sample_list[i]:
                count += 1
        count = count * 2
        tmp_distance = tmp_distance - count
        tmp_distance1 = math.pow(float(tmp_distance),0.5)
        distance[i] = tmp_distance1

    return distance

def find_mode1(label_list,weight_coefficient,sorted_data):
    fre = {}
    size = len(sorted_data)

    # 遍历，统计各个label出现的次数
    for i in range(size):
        index = sorted_data[i][0]
        if sorted_data[i][1] == 0:  # 如果距离为0，说明两个句子完全吻合，赋予极大权值
            weight = 10000
        else:
            weight = 1 / (sorted_data[i][1])  # 赋予距离倒数，距离越小，倒数越大
        emotion = label_list[index]
        if emotion in fre.keys():
            # 为了让距离更小的有更大的权值，对当前的权值进行幂次运算，增大权重
            fre[emotion] += math.pow(weight, weight_coefficient)
        else:
            fre[emotion] = math.pow(weight, weight_coefficient)

    max_value = max(fre.values())  # 找出权重最大的label
    for key, value in fre.items():
        if value == max_value:  # 返回权重最大的label
            return key


def main():

    # file_name = 'classification/train_set.csv'
    # with open(file_name,'r',encoding='utf-8') as csvfile:  #打开训练集csv文件
    #     csv_reader = csv.reader(csvfile)
    #     sample_list = list(csv_reader)   #将数据转为列表，方便计算

    file_name = '2/trainData.txt'
    with open(file_name,'r',encoding='utf-8') as file1:
        sample_list = file1.readlines()

    # file_name1 = 'exam_test/classification_simple_test.csv'
    # with open(file_name1,'r',encoding='utf-8') as csvfile1:   #打开验证集文件
    #     csv_reader1 = csv.reader(csvfile1)
    #     validation_list = list(csv_reader1)   #转为列表

    file_name1 = '2/trainData.txt'
    with open(file_name1,'r',encoding = 'utf-8') as file2:
        validation_list = file2.readlines()

    file_name2 = '2/trainLabel.txt'
    with open(file_name2,'r',encoding = 'utf-8') as file3:
        label_list = file3.readlines()

    validation_size = len(validation_list)  #验证集长度
    size1 = len(sample_list)
    label_size = len(label_list)

    for i in range(size1):
        sample_list[i] = list(set(sample_list[i].split()))

    for i in range(validation_size):
        validation_list[i] = list(set(validation_list[i].split()))

    word_list = []
    #找出不重复词汇列表
    for i in range(size1):
        tmp_list = sample_list[i][0].split()
        size2 = len(tmp_list)
        for j in range(size2):
            if tmp_list[j] not in word_list:
                word_list.append(tmp_list[j])

    size3 = len(word_list)
    # print(size3)
    one_hot = build_one_hot(sample_list,word_list)  #建立训练集的onehot矩阵
    validation_one_hot = build_validation_one_hot(validation_list,word_list)    #建立验证集的onehot矩阵


    #权重系数
    weight_coefficient1 = 2
    #选择哪种衡量方法
    measure_choice1  = [0]
    #k值
    # knn_value = [11,12,13,14,15,16,17,18,19,20]
    knn_value = [13]
    # knn_value = [22]
    csv_file2 = open('2/16337098.csv', 'w', newline="")  #打开要写入的文件
    writer = csv.writer(csv_file2)
    #对应每个权值系数进行计算
    for measure_choice in measure_choice1:
        #对应每个k值进行计算
        for k_value in knn_value:
            writer.writerow(['Words (split by space)','label'])  #输入标题
            correct = 0
            # start = time.clock()
            for x in range(validation_size):   #遍历每一行验证集的句子

                # sorted_data = measure(one_hot,validation_one_hot,x,size1,measure_choice)

                # test_emotion = find_mode(sample_list,sorted_data[:k_value],weight_coefficient1,measure_choice)   #找出选出的k个情感中的众数，得出最终的情感

                distance = get_distance(sample_list,validation_list,x)
                sorted_data = sorted(distance.items(), key=operator.itemgetter(1))  # 距离排序
                test_emotion = find_mode1(label_list,weight_coefficient1,sorted_data[:k_value])

                if label_list[x] == test_emotion:  #计算准确度
                    correct += 1
                writer.writerow([validation_list[x][0],test_emotion])
                print(label_list[x] == test_emotion)

            print("k w m : ",str(k_value),str(weight_coefficient1),str(measure_choice),"   correct rate: ",str(correct / validation_size))
            # end = time.clock()
            # print("run time:  ", str(end - start), " s")
            # print("------")


if __name__ == "__main__":
    main()

