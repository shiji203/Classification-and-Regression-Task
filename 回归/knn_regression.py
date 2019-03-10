import csv
import math
import time
import operator
import numpy as np
import scipy.spatial.distance as dist
import pandas

#建立验证集onehot矩阵
def build_validation_one_hot(validation_list,word_list):
    validation_one_hot = []  #onehot矩阵
    validation_size = len(validation_list)  #验证集大小
    word_size = len(word_list)   #无重复词语列表大小

    #c初始化验证集列表
    for i in range(validation_size):
        tmp_list = [0] * word_size
        validation_one_hot.append(tmp_list)

    #构造onehot矩阵
    for i in range(validation_size):
        # validation_list[i] = validation_list[i][1:]
        validation_tmp_list = validation_list[i][0].split() #取出前面的句子
        validation_tmp_list_size = len(validation_tmp_list)  #句子大小
        for j in range(validation_tmp_list_size):   #对句子的每个词语操作
            for k in range(word_size):
                if validation_tmp_list[j] == word_list[k]:  #找到词语在word_list中的位置
                    validation_one_hot[i][k] = 1   #在onehot对应的位置置1
                    break

    return validation_one_hot

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
        a1 = np.array(one_hot[i])  #转换为array
        a2 = np.array(validation_one_hot[x])
        distance[i] = np.linalg.norm(a1 - a2, ord=1)   #应用公式求街区距离
    return distance

#求欧氏距离
def get_euclidean_distance(one_hot,validation_one_hot,x,sample_list_size):

    distance = {}
    for i in range(sample_list_size):
        # dis_tmp = 0
        # for j in range(word_list_size):
        #     dis_tmp += math.pow((one_hot[i][j] - validation_one_hot[x][j]), 2)
        # dis_tmp = math.pow(dis_tmp, 0.5)
        a1 = np.array(one_hot[i])   #转换为array，便于下一步计算
        a2 = np.array(validation_one_hot[x])
        dis_tmp = np.linalg.norm(a1 - a2)   #用求二范数公式
        distance[i] = dis_tmp
    return distance

#求汉明距离
def get_hanming_distance(one_hot,validation_one_hot,x,sample_list_size):
    distance = {}
    for i in range(sample_list_size):
        a1 = np.array(one_hot[i])   #转换为array
        a2 = np.array(validation_one_hot[x])
        smstr = np.nonzero(a1 - a2)
        distance[i] = np.shape(smstr[0])[0]
    return distance

#求杰卡德距离
def get_jake_distance(one_hot,validation_one_hot,x,sample_list_size):
    distance = {}
    for i in range(sample_list_size):
        a1 = np.array(one_hot[i])
        a2 = np.array(validation_one_hot[x])
        matv = np.array([a1,a2])
        distance[i] = dist.pdist(matv,'jaccard')
    return distance

#余弦相似度
def get_cosine_similarity(train_array,test_array,x,sample_list_size):
    cosine_similarity = {}
    for i in range(sample_list_size):
        m = np.linalg.norm(train_array[i]) * (np.linalg.norm(test_array[x]))
        if (m != 0):
            cosine_similarity[i] = np.dot(train_array[i], test_array[x]) / m   #应用余弦公式
        else:
            cosine_similarity[i] = 0
    return cosine_similarity

#求相关系数
def get_correlation_coefficient(list_a,list_b):
        s1 = pandas.Series(list_a)
        s2 = pandas.Series(list_b)
        corr = s1.corr(s2)
        return corr

#确定采用哪种计算距离方式
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

#得到各种情绪的概率
def get_final_emotion(train_array,sorted_data,k_value,weight_coefficient,measure_choice):
    test_score = 0.0
    #余弦相似度的求解方法
    if measure_choice == 2:
        sum1 = 0
        for i in range(k_value):
            sum1 += math.pow((sorted_data[i][1] * 15 + 1), weight_coefficient) #求出总权值
            # sum1 += math.pow(math.e,(sorted_data[i][1]*15+1))
        for i in range(k_value):
            index = sorted_data[i][0]  #得到下标
            min = math.pow((sorted_data[i][1] * 15 + 1), weight_coefficient)
            # min = math.pow(math.e,(sorted_data[i][1]*15+1))
            weight = min / sum1  #权值占比
            test_score += train_array[index][-1]*weight

    # #距离的求解方法
    # else:
    #     sum1 = 0
    #     for i in range(k_value):
    #         sum1 += math.pow(1 / (sorted_data[i][1] + 1), weight_coefficient)  #距离倒数，为了增强距离近的比重，加入幂次运算，增大权值
    #     for i in range(k_value):
    #         index = sorted_data[i][0]
    #         min = math.pow(1 / (sorted_data[i][1] + 1), weight_coefficient)
    #         weight = min / sum1   #权值占比
    #         anger += float(sample_list[index][1]) * weight   #依次乘以权值占比
    #         disgust += float(sample_list[index][2]) * weight
    #         fear += float(sample_list[index][3]) * weight
    #         joy += float(sample_list[index][4]) * weight
    #         sad += float(sample_list[index][5]) * weight
    #         surprise += float(sample_list[index][6]) * weight

    return test_score

def main():
    start = time.clock()
    file_name = 'train.csv'  #打开训练集
    with open(file_name,'r',encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        file1_list = list(csv_reader)   #训练集数据列表

    file1_list.pop(0)
    train_list = file1_list[0:60000]
    test_list = file1_list[60000:80000]

    train_array = np.array(train_list)
    train_array = np.delete(train_array, 6, axis=1)
    train_array = np.array(train_array, dtype=float)
    test_array = np.array(test_list)
    test_array = np.delete(test_array, 6, axis=1)
    test_array = np.array(test_array, dtype=float)

    train_size = len(train_array)
    test_size = len(test_array)

    for i in range(6):
        word_list = train_array[:, i]
        max_value = max(word_list)
        min_value = min(word_list)
        ran = max_value - min_value
        word_size = len(word_list)
        for k in range(word_size):
            word_list[k] = float(word_list[k] - min_value) / ran
        train_array[:, i] = word_list

    for i in range(6):
        word_list = test_array[:, i]
        max_value = max(word_list)
        min_value = min(word_list)
        ran = max_value - min_value
        word_size = len(word_list)
        for k in range(word_size):
            word_list[k] = float(word_list[k] - min_value) / ran
        test_array[:, i] = word_list


    weight_coefficient = 5  #权重系数
    measure_choice = 2   #度量方式选择
    knn_value = [150]
    score_list = []
    test_score_list = []
    #对每个k值进行遍历
    for k_value in knn_value:
        for x in range(test_size):   #遍历每个句子
            print(x+1)
            sorted_data = measure(train_array,test_array,x,train_size,measure_choice)   #采用对应的测量方式得到最佳的k个样本
            test_score = get_final_emotion(train_array,sorted_data,k_value,weight_coefficient,measure_choice)  #得到预测样本的各个概率
            test_score_list.append(test_score) #预测结果列表
            score_list.append(float(test_array[x][-1])) #标准结果列表
            print(abs(test_score - test_array[x][-1])) #计算误差，观察预测过程误差范围

    corr = get_correlation_coefficient(test_score_list,score_list)
    print("相关系数： ",str(corr))
    end = time.clock()
    print("run time:  ", str(end - start), " s")


if __name__ == "__main__":
    main()











