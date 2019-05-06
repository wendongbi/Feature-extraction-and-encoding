# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:18:20 2019

@author: Damon
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

####hyperparameter
Size_of_word=64

def load_data():
    train_data  = np.load("../sift_data/data_train.npy" )
    train_label = np.load("../sift_data/label_train.npy")
    test_data   = np.load("../sift_data/data_test.npy")
    test_label  = np.load("../sift_data/label_test.npy")
    
    #train_data = train_data.reshape(train_data.shape[0],-1,Size_of_word)
    #test_data  =  test_data.reshape( test_data.shape[0],-1,Size_of_word)
#    train_data  =  train_data[:100]
#    train_label = train_label[:100]
#    test_data   =   test_data[:100]
#    test_label  =  test_label[:100]
    
    return train_data, train_label, test_data, test_label


"""
功能：提取所有图像的SURF特征条形图统计量
输入：
    img_descs：提取的SURF特征
    cluster_model: 未训练的聚类模型
输出：
    img_bow_hist：条形图，即最终的特征
    cluster_model：训练好的聚类模型
"""
def cluster_features(img_descs, cluster_model):
    n_clusters = cluster_model.n_clusters #要聚类的种类数
    #将所有的特征排列成N*D的形式，其中N表示特征数，
    #D表示特征维度，这里特征维度D=64
    train_descs = [desc for desc_list in img_descs
                       for desc in desc_list]
    print("length of train_descs", len(train_descs))
    print(type(train_descs[0]), train_descs[0].shape)
    train_descs = np.array(train_descs)#转换为numpy的格式
    print(train_descs.shape, train_descs.shape[1])
    
    #判断D是否为64
    if train_descs.shape[1] != Size_of_word: 
        raise ValueError('期望的SURF特征维度应为{}'.format(Size_of_word) +'  实际为'
                         , train_descs.shape[1])
    #训练聚类模型，得到n_clusters个word的字典
    cluster_model.fit(train_descs)
    #raw_words是每张图片的SURF特征向量集合，
    #对每个特征向量得到字典距离最近的word
    img_clustered_words = [cluster_model.predict(raw_words)
                           for raw_words in img_descs]
    #print(len(img_clustered_words))
    #对每张图得到word数目条形图(即字典中每个word的数量)
    #即得到我们最终需要的特征
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters)
         for clustered_words in img_clustered_words])
    print((img_bow_hist.shape))
    np.save("../data_process/BOF_Size_of_word{}".format(Size_of_word)+
            "_component{}.npy".format(cluster_model.n_clusters), img_bow_hist)
    
    return img_bow_hist, cluster_model

def run_svm(train_datas, train_labels):   
    classifier = OneVsRestClassifier(
        LinearSVC(random_state=0)).fit(train_datas, train_labels)
    return classifier


def test(test_datas, test_labels, cluster_model, classifier):
    print ("测试集的数量: ", len(test_datas))
    preds = []
    for test_data in test_datas:
        clustered_desc = cluster_model.predict(test_data)
        img_bow_hist = np.bincount(clustered_desc,
                               minlength=cluster_model.n_clusters)
        #转化为1*K的形式,K为字典的大小，即聚类的类别数
        vect = img_bow_hist.reshape(1,-1)
        pred = classifier.predict(vect)
        preds.append(pred[0])
    preds = np.array(preds)
    idx = preds == test_labels
    accuracy = sum(idx)/len(idx)
    print ("Accuracy is : ", accuracy)
    
    return accuracy



#train_datas, train_labels, test_datas, test_labels = load_data()
####hyperparameter
Ks = [500] #要聚类的数量，即字典的大小(包含的单词数)

for K in Ks:
    train_datas, train_labels, test_datas, test_labels = load_data()
    cluster_model=MiniBatchKMeans(n_clusters=K, init_size=3*K)
    train_datas, cluster_model = cluster_features(train_datas,
                                                  cluster_model)
    
    
    #将训练集label转化为numpy.array类型
    train_labels = np.array(train_labels)
    classifier = run_svm(train_datas, train_labels)
    
    test_labels = np.array(test_labels)
    accuracy = test(test_datas, test_labels, cluster_model, classifier)
    with open("../BOF_result.txt",'a') as f:
        f.write("Size_of_word ={} ".format(Size_of_word)+
                "K = {}".format(K)+  "  Accuracy: {}\n".format(accuracy))


