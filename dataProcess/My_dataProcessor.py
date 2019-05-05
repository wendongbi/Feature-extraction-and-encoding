import numpy as np
import csv
from PIL import Image
import os
import glob
import shutil

class_map = {
    'pb': 0,
    'i13': 1,
    'p19': 2,
    'pm20': 3,
    'w55': 4,
    'w42': 5,
    'p10': 6,
    'p1': 7,
    'p5': 8,
    'il80': 9,
    'ip': 10,
    'pn': 11,
    'ps': 12,
    'pr30': 13,
    'p9': 14,
    'p23': 15,
    'pl40': 16,
    'w62': 17,
    'pl100': 18,
    'pl120': 19,
    'pr40': 20,
    'i3': 21,
    'i10': 22,
    'pg': 23,
    'p24': 24,
    'i9': 25,
    'w63': 26,
    'pl50': 27,
    'pne': 28,
    'w43': 29,
    'nn': 30
}

# srcPath1 = './Raw/Total/'
# srcPath2 = './Raw/test/'
# savePath = './awa2/Total/'
# folderList1 = os.listdir(savePath)
# folderList2 = os.listdir(srcPath2)

# # rename file
# folderList1 = os.listdir(savePath)
# for i in range(len(folderList1)):
#     if os.path.exists(savePath + folderList1[i]):
#         os.rename(savePath + folderList1[i], savePath + str(class_map[folderList1[i]]))
#     else:
#         print('does not exiist.')



# # clamp two dataset
# count = 0
# for i in range(len(folderList2)):
#     if os.path.exists(srcPath2 + folderList2[i]):
#         print('begin saving ' + folderList2[i])
        
#         for imgFile in glob.glob(srcPath2 + folderList2[i] + '/*.jpg'):

#             targetFile = savePath + str(class_map[folderList2[i]]) + '/' + imgFile.split('/')[-1]
#             # shutil.copy(imgFile, targetFile)
#             img = Image.open(imgFile)
#             img.save(targetFile, quality=95)
#             print(targetFile)
#             count += 1
#             print('save sucess')
# print(count)


src_path = './raw/'

train_path = './my_data2/train_data/'
test_path = './my_data2/test_data/'
total_path = './my_data2/Total_data/'

if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)
if not os.path.exists(total_path):
    os.makedirs(total_path)

train_csv =  open('train2.csv','a', newline='')
test_csv = open('test2.csv','a', newline='')
total_csv = open('total2.csv', 'a', newline='')
classMap_csv = open('classMap.csv', 'a', newline='')
train_csv_writer = csv.writer(train_csv, dialect='excel')
test_csv_writer = csv.writer(test_csv, dialect='excel')
total_csv_writer = csv.writer(total_csv, dialect='excel')
classMap_csv_writer = csv.writer(classMap_csv, dialect='excel')

folderList = os.listdir(src_path)
print(len(folderList))

i = 0
for class_name in folderList:
    imgList = os.listdir(src_path + class_name + '/')
    print(class_name, ':' ,len(imgList))
    j = 0
    classMap_csv_writer.writerow([class_name, i])
    for imgFile in glob.glob(src_path + class_name + '/*.jpg'):
        img = Image.open(imgFile)
        imgName = imgFile.split('/')[-1]
        csv_item = [imgName, i]
        if j < 8 :
            img.save(train_path + imgName, quality=95)
            train_csv_writer.writerow(csv_item)
        else:
            img.save(test_path + imgName, quality=95)
            test_csv_writer.writerow(csv_item)
        img.save(total_path + imgName, quality=95)
        total_csv_writer.writerow(csv_item)
        print('save success')
        j = (j + 1) % 10
    i += 1