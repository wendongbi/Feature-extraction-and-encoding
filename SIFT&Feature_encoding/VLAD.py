# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 10:43:18 2019

@author: Damon
"""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA as sklearnPCA

def load_data():
#    train_data  = np.load("../sift_data/data_train.npy" )
#    train_label = np.load("../sift_data/label_train.npy")
#    test_data   = np.load("../sift_data/data_test.npy")
#    test_label  = np.load("../sift_data/label_test.npy")
    
    train_data  = np.load("../d_data_new/data_train.npy" )
    train_label = np.load("../d_data_new/label_train.npy")
    test_data   = np.load("../d_data_new/data_test.npy" )
    test_label  = np.load("../d_data_new/label_test.npy" )
    print("Load Data Done=========================")
    print(train_data.shape)
    print(train_label.shape)
    print(test_data.shape)
    print(test_label.shape)
    #train_data = train_data.reshape(train_data.shape[0],-1,DESDIM)
    #test_data  =  test_data.reshape( test_data.shape[0],-1,DESDIM)
    
    return train_data, train_label, test_data, test_label 

  
def get_des_vector(train_datas):
    #length1 = train_datas.shape[0]
    #length2 = train_datas.shape[1]
    #all_des = np.empty(shape=[0, DESDIM])
    #for i,each_des in enumerate(train_datas):
        #print("i",i)
    #    all_des = np.concatenate([all_des, each_des])
    
    all_des = np.load("../deep_data/proposal_feature.npy")
    
    image_des_len = np.genfromtxt('../deep_data/proposal_num.txt',
                                  dtype=np.uint8, max_rows=37321, skip_header=0)
    #image_des_len = [train_datas[i].shape[0] for i in range(length1)]
    print(len(image_des_len),image_des_len[0],image_des_len[1])
    print(all_des.shape)
    
    return all_des, image_des_len

def get_cluster_center(des_set, K):
    '''
    Description: cluter using a default setting
    Input: des_set - cluster data
                 K - the number of cluster center
    Output: laber  - a np array of the nearest center for each cluster data
            center - a np array of the K cluster center
    '''
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)
    des_set = np.float32(des_set)
    ret, label, center = cv2.kmeans(des_set, K, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
    return label, center

def get_codebook(all_des, K):
    '''
    Description: train the codebook from all of the descriptors
    Input: all_des - training data for the codebook
                 K - the column of the codebook

    '''
    label, center = get_cluster_center(all_des, K)
    print(label.shape)
    print(center.shape)
    return label, center

def get_vlad_base(img_des_len, NNlabel, all_des, codebook):
    '''
    Description: get all images vlad vector 
    '''
    cursor = 0
    vlad_base = []
    for eachImage in img_des_len:
        descrips = all_des[cursor : cursor + eachImage]
        centriods_id = NNlabel[cursor : cursor + eachImage]
        centriods = codebook[centriods_id]
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
        cursor += eachImage
    
        vlad_norm = vlad.copy()
        cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
        vlad_base.append(vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1))
    
    print("get_vlad_base========")    
    print(len(vlad_base),len(vlad_base[0]),len(vlad_base[1]))
    vlad_base = np.array(vlad_base)
    print(vlad_base.shape)
    vlad_base = vlad_base.reshape(vlad_base.shape[0],-1)
    print(vlad_base.shape)
    return vlad_base

def get_vlad_base_pca(img_des_len, NNlabel, all_des, codebook):
    cursor = 0
    vlad_base = []
    for eachImage in img_des_len:
        descrips = all_des[cursor : cursor + eachImage]
        centriods_id = NNlabel[cursor : cursor + eachImage]
        centriods = codebook[centriods_id]
    
        vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
        for eachDes in range(eachImage):
            vlad[centriods_id[eachDes]] = vlad[centriods_id[eachDes]] + descrips[eachDes] - centriods[eachDes]
        cursor += eachImage

        vlad_base.append(vlad.reshape(COLUMNOFCODEBOOK * DESDIM, -1))
    
    vlad_base_pca = np.array(vlad_base)
    vlad_base_pca = vlad_base_pca.reshape(-1, DESDIM * COLUMNOFCODEBOOK)
    sklearn_pca = sklearnPCA(n_components=PCAD)
    sklearn_transf = sklearn_pca.fit_transform(vlad_base_pca)
    sklearn_transf_norm = sklearn_transf.copy()
    for each, each_norm in zip(sklearn_transf, sklearn_transf_norm):
        cv2.normalize(each, each_norm, 1.0, 0.0, cv2.NORM_L2)
    print("sklearn_transf_norm, shape : ",sklearn_transf_norm.shape)
    return sklearn_transf_norm, sklearn_pca



def run_svm(train_datas, train_labels):   
    classifier = OneVsRestClassifier(
        LinearSVC(random_state=0)).fit(train_datas, train_labels)
    return classifier



def cal_vec_dist(vec1, vec2):
    '''
    Description: calculate the Euclidean Distance of two vectors
    '''
    return np.linalg.norm(vec1 - vec2)
def get_pic_vlad(pic, des_size, codebook):
    '''
    Description: get the vlad vector of each image
    '''
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = float("inf")
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind]
    
    vlad_norm = vlad.copy()
    cv2.normalize(vlad, vlad_norm, 1.0, 0.0, cv2.NORM_L2)
    vlad_norm = vlad_norm.reshape(COLUMNOFCODEBOOK * DESDIM, -1)
    vlad_norm = np.array(vlad_norm)
    #print(vlad_norm.shape)
    
    return vlad_norm

def get_pic_vlad_pca(pic, des_size, codebook, sklearn_pca):
    vlad = np.zeros(shape=[COLUMNOFCODEBOOK, DESDIM])
    for eachDes in range(des_size):
        des = pic[eachDes]
        min_dist = float("inf")
        ind = 0
        for i in range(COLUMNOFCODEBOOK):
            dist = cal_vec_dist(des, codebook[i])
            if dist < min_dist:
                min_dist = dist
                ind = i
        vlad[ind] = vlad[ind] + des - codebook[ind]
    
    vlad = vlad.reshape(-1, COLUMNOFCODEBOOK * DESDIM)
    sklearn_transf = sklearn_pca.transform(vlad)
    sklearn_transf_norm = sklearn_transf.copy()
    cv2.normalize(sklearn_transf, sklearn_transf_norm, 1.0, 0.0, cv2.NORM_L2)
    #print(sklearn_transf_norm.shape)
    
    return sklearn_transf_norm


def test(test_des, test_des_len, test_labels, codebook, classifier):
    print ("测试集的数量: ", len(test_des_len))
    preds = []
    
    cursor_test = 0
    
    for eachtestpic in range(len(test_des_len)):
        pic = test_des[cursor_test:cursor_test+test_des_len[eachtestpic]]
        test_vlad = get_pic_vlad(pic, test_des_len[eachtestpic],codebook)
        cursor_test += test_des_len[eachtestpic]

        vect = test_vlad.reshape(1,-1)
        pred = classifier.predict(vect)
        preds.append(pred[0])
    preds = np.array(preds)
    idx = preds == test_labels
    accuracy = sum(idx)/len(idx)
    print ("Accuracy is : ", accuracy)
    
    return accuracy

def test_pca(test_des, test_des_len, test_labels, codebook, classifier,sklearn_pca):
    print ("测试集的数量: ", len(test_des_len))
    preds = []
    
    cursor_test = 0

    for eachtestpic in range(len(test_des_len)):
        pic = test_des[cursor_test:cursor_test+test_des_len[eachtestpic]]
        test_vlad_pca = get_pic_vlad_pca(pic, test_des_len[eachtestpic],codebook,sklearn_pca)
        cursor_test += test_des_len[eachtestpic]
        
        vect = test_vlad_pca.reshape(1,-1)
        pred = classifier.predict(vect)
        preds.append(pred[0])
    preds = np.array(preds)
    idx = preds == test_labels
    accuracy = sum(idx)/len(idx)
    print("Accuracy is : ", accuracy)
    
    return accuracy

####hyperparameter
COLUMNOFCODEBOOKs = [20]
DESDIM = 2048
PCAD = 128
TESTTYPE=1




for COLUMNOFCODEBOOK in COLUMNOFCODEBOOKs:
    train_datas, train_labels, test_datas, test_labels = load_data()
    all_des, image_des_len = get_des_vector(train_datas)
    test_des, test_des_len = get_des_vector(test_datas)
    ##trainning the codebook
    NNlabel, codebook = get_codebook(all_des, COLUMNOFCODEBOOK)


    ####if testtype == 0,vlad only
    if TESTTYPE == 0: 
    
        vlad_base = get_vlad_base(image_des_len, NNlabel, all_des, codebook)
        classifier = run_svm(vlad_base, train_labels)
    
        ##get all the vlad vectors of retrival set without pca dimensionality reduction
        test_labels = np.array(test_labels)
        accuracy = test(test_des, test_des_len,test_labels, codebook, classifier)
    #    cursor_ret = 0
    #    ret_vlad_list = []
    #    for eachretpic in range(len(ret_des_len)):
    #        pic = ret_des[cursor_ret: cursor_ret + ret_des_len[eachretpic]]
    #        ret_vlad = get_pic_vlad(pic, ret_des_len[eachretpic], codebook)
    #        cursor_ret += ret_des_len[eachretpic]
    #        ret_vlad_list.append(ret_vlad)
        with open("..\VLDA_result.txt",'a') as f:
            f.write("Size_of_word ={} ".format(DESDIM)+
                    "K = {}".format(COLUMNOFCODEBOOK)+  "  Accuracy: {}\n".format(accuracy))
    
    if TESTTYPE == 1:
        vlad_base_pca, sk_pca = get_vlad_base_pca(image_des_len, NNlabel, all_des, codebook)
        classifier = run_svm(vlad_base_pca, train_labels)
        
        ##get all the vlad vectors of retrival set with pca dimensionality reduction
        test_labels = np.array(test_labels)
        accuracy = test_pca(test_des, test_des_len,test_labels, codebook, classifier,sk_pca)    
        with open("..\VLDA_result.txt",'a') as f:
            f.write("PCA_Dimension={}".format(PCAD) + "Size_of_word ={} ".format(DESDIM)+
                    "K = {}".format(COLUMNOFCODEBOOK)+  "  Accuracy: {}\n".format(accuracy))    
    
    
    
    

