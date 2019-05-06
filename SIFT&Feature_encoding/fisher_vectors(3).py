import sys, argparse
import numpy as np
from sklearn.svm import SVC
from sklearn import mixture
import math
import time
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from functools import reduce
import multiprocessing

def load_data():
    # train_data = np.genfromtxt(fname='./Data/data_train.txt', dtype=np.float32, skip_header=0)
    # print(train_data.shape)
    # test_data = np.genfromtxt(fname='./Data/data_test.txt', dtype=np.float32, skip_header=0)
    # print(test_data.shape)
    # train_label = np.genfromtxt(fname='./Data/label_train.txt', dtype=np.float32, skip_header=0)
    # print(train_label.shape)
    # test_label = np.genfromtxt(fname='./Data/label_test.txt', dtype=np.float32, skip_header=0)
    # print(test_label.shape)
    # return train_data, test_data, train_label, test_label
    train_data = np.load("./Data/sift_data_mean/data_train.npy")
    print(train_data.shape)
    print(train_data[0].shape)
    train_label = np.load("./Data/sift_data_mean/label_train.npy")
    print(train_label.shape)
    test_data = np.load("./Data/sift_data_mean/data_test.npy")
    print(test_data.shape)
    print(test_data[0].shape)
    test_label = np.load("./Data/sift_data_mean/label_test.npy")
    print(test_label.shape)

    return train_data, test_data, train_label, test_label

def dictionary(descriptors, N):
    gmm = mixture.GaussianMixture(n_components=N)
    gmm.fit(descriptors)
    return np.float32(gmm.means_),np.float32(gmm.covariances_ ),np.float32(gmm.weights_)

def likelihood_moment(x, ytk, moment):
    x_moment = np.power(np.float32(x), moment) if moment > 0 else np.float32([1])
    return x_moment * ytk

def likelihood_statistics(samples, means, covs, weights):
    s0, s1,s2 = {}, {}, {}
    sample = zip(range(0, samples.shape[0]), samples)
    sample1 = zip(range(0, samples.shape[0]), samples)
    gaussians = {}

    g = [multivariate_normal(mean=means[k], cov=covs[k], allow_singular = True) for k in range(0, weights.shape[0]) ]
    for i,x in sample:
        gaussians[i] = np.array([g_k.pdf(x) for g_k in g])

    for k in range(0, weights.shape[0]):
        s0[k], s1[k], s2[k] = 0, 0, 0
        for i,x in sample1:

            probabilities = np.multiply(gaussians[i],weights)
            if (np.sum(probabilities) != 0):
                probabilities = probabilities/np.sum(probabilities)

            s0[k] = s0[k] + likelihood_moment(x, probabilities[k], 0)
            s1[k] = s1[k] + likelihood_moment(x, probabilities[k], 1)
            s2[k] = s2[k] + likelihood_moment(x, probabilities[k], 2)

    return s0, s1, s2

def fisher_vector_weights(s0, s1, s2, means, covs, w, T):
    return np.float32([((s0[k] - T * w[k]) / np.sqrt(w[k]) ) for k in range(0, w.shape[0])])

def fisher_vector_means(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s1[k] - means[k] * s0[k]) / (np.sqrt(w[k] * sigma[k])) for k in range(0, w.shape[0])])

def fisher_vector_sigma(s0, s1, s2, means, sigma, w, T):
    return np.float32([(s2[k] - 2 * means[k]*s1[k]  + (means[k]*means[k] - sigma[k]) * s0[k]) / (np.sqrt(2*w[k])*sigma[k])  for k in range(0, w.shape[0])])

def normalize(fisher_vector):
    v = np.sqrt(abs(fisher_vector)) * np.sign(fisher_vector)
    if(np.sqrt(np.dot(v, v)) == 0):
        return v
    else:
        return v / np.sqrt(np.dot(v, v))

def generate_gmm(gmm_folder,descriptors, N):
    '''
    Interface
    gmm_folder
    descriptors, (Train data) ,numpy.array, matrix, each row is one sample
    N, int ,the number of cluster center
    '''
    print("start generate GMM")
    means,covs,weights = dictionary(descriptors,N)
    # np.save(gmm_folder+ '/'+ N + "means.gmm", means)
    # np.save(gmm_folder+ '/'+ N + "covs.gmm", covs)
    # np.save(gmm_folder+ '/'+ N + "weights.gmm", weights)
    return means, covs, weights

def load_gmm(folder = "."):
    '''
    Interface
    '''
    files = ["means.gmm.npy", "covs.gmm.npy", "weights.gmm.npy"]
    return map(lambda file: np.load(open(file,'rb')), map(lambda s : folder + "/" + s , files))

def fisher_vector(samples, means, covs, w):
    '''
    Interface:
    samples: (to be en
    coded ),numpy.array , matrix, each row is a sample
    means: gmm.means_
    covs: gmm.covars_
    w: gmm.weights_
    '''
    s0, s1, s2 =  likelihood_statistics(samples, means, covs, w)
    T = samples.shape[0]
    covs = np.float32([np.diagonal(covs[k]) for k in range(0, covs.shape[0])])
    # a = concatenate(fisher_vector_weights(s0, s1, s2, means, covs, w, T))
    # a = fisher_vector_weights(s0, s1, s2, means, covs, w, T)
    b = fisher_vector_means(s0, s1, s2, means, covs, w, T)
    c = fisher_vector_sigma(s0, s1, s2, means, covs, w, T)
    fv = np.concatenate([np.concatenate(b),np.concatenate(c)])
    fv = normalize(fv)
    return fv

if __name__ == '__main__':
    train_data, test_data, train_label, test_label = load_data()

    train_data = np.array(train_data)
    test_data = np.array(test_data)

    all_desc = np.empty(shape = [0,128])
    for i, each_desc in enumerate(train_data):
        print(i)
        print(each_desc.shape)
        len = int(each_desc.shape[0] * 0.05)
        all_desc = np.concatenate([all_desc,each_desc[:len]])
    print(all_desc.shape)

    for k in [10]:
        print(k)
        means, covs, weights = generate_gmm(gmm_folder="./GMM", descriptors=all_desc, N=k)
        # means, covs, weights = load_gmm("./GMM")

        new_train_data = []
        new_test_data = []
        for i in range(train_data.shape[0]):
            print("train data"+str(i))
            fv = fisher_vector(samples = train_data[i] ,means = means, covs = covs, w = weights)
            new_train_data.append(fv)
        new_train_data = np.array(new_train_data)
        print(new_train_data.shape)

        for j in range(test_data.shape[0]):
            print("test_data"+str(j))
            fv = fisher_vector(samples = test_data[j] ,means = means, covs = covs, w = weights)
            new_test_data.append(fv)
        new_test_data = np.array(new_test_data)
        print(new_test_data.shape)

        np.save("./feature_encoding/FV_train_sift_mean_k{}".format(k), new_train_data)
        np.save("./feature_encoding/FV_test_sift_mean_k{}".format(k), new_test_data)

