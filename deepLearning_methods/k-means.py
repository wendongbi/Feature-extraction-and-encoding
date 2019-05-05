import numpy as np
from sklearn.cluster import KMeans
# feature = np.load('feature3.npy').tolist()
# # print(np.shape(feature))
# label = np.load('./label3.npy').tolist()
# train = []
# test = []
# label_train = []
# label_test = []
# count = 0
# for i in range(len(label)):
#     if(count < 6):
#         train.append(feature[i])
#         label_train.append(label[i])
#         count = (count + 1) % 10
#     else:
#         test.append(feature[i])
#         label_test.append(label[i])
#         count = (count + 1) % 10
# print(np.shape(train))
# print(np.shape(test))
# print(np.shape(label_train))
# print(np.shape(label_test))
# np.save('./train_set.npy', np.array(train))
# np.save('./train_label.npy', np.array(label_train).reshape(-1))
# np.save('./test_set.npy', np.array(test))
# np.save('./test_label.npy', np.array(label_test).reshape(-1))

# labels = []

# clf = KMeans(n_clusters=50, init='k-means++')
# dataset = np.load('./feature3.npy')
# label = np.load('./label3.npy')

# test_data = dataset[:100, :]
# clf.fit(dataset)
# test_labels = clf.labels_
# test_labels = np.array(test_labels).reshape(-1, 1)
# np.savetxt('./k-means_result.txt', test_labels)

# z = np.zeros(50).tolist()
# print(z)
feature = [30]
result = np.genfromtxt('./k-means_result.txt')
# for i in range(len(result)):
#     tmp = np.zeros(50).tolist()
#     tmp[int(result[i])] = 1
#     feature.append(tmp)
# np.save('./k_feature.npy', np.array(feature))


def showmax(lt):
    index = 0
    max = 0
    for i in range(len(lt)):
        flag = 0
        for j in range(i + 1, len(lt)):
            if(lt[j] == lt[i]):
                flag+=1
        if flag>max:
            max = flag
            index = i
    return lt[index]

k_feature = np.load('./k_feature.npy')
output = np.argmax(k_feature, axis=1)
label = np.load('./label3.npy').tolist()
# idx = 0
# count = np.zeros(50, dtype=int)
# for i in range(len(label)):
#     if label[i] == idx:
#         count[idx] += 1
#     else:
#         idx = label[i]
#         count[idx] += 1
# print(np.sum(count))
# print(count)

count = [1644, 1019,  100, 1202,  872 , 704,  512,  664, 1420,  185,  895,  988,  193,  272,
  500,  567 , 728 ,1170 , 383, 1200,  758,  720, 1344,  188 , 630 , 946,  174 ,1088,
  709,  291 , 877  ,310 ,1038,  215, 1046 , 589  ,713,  852 , 779 ,1338,  747 ,1027,
  549 , 874  ,291 , 868, 1033 , 696,  684,  728]
start = 0
end = 0
num = 0
test = []
correct = np.zeros(50)
for i in range(len(count)):
    start = end
    end = start + count[i]
    out_label = showmax(output[start:end])
    for j in range(end-start):
        if(output[start+j] == out_label):
            correct[i] += 1
print(correct)
accu = np.sum(correct)/len(output)
print('accuracy:', accu)
class_accu = correct
for i in range(len(class_accu)):
    class_accu[i] /= count[i]
print(class_accu)
x = range(len(class_accu))
y = class_accu
import matplotlib.pyplot as plt
plt.bar(x, y, color='red')
plt.xlabel('class idx')
plt.ylabel('accu')
plt.title('k-means(Total accu='+str(accu)+')')
plt.show()


