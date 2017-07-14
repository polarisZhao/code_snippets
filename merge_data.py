#-*- coding: utf-8 -*-
import numpy as np
from PIL import Image

def make_arrays(nb_rows, img_height, img_width):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_height, img_width), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels

def merge_datasets(path, img_height,img_width):
    '''
       func: 将文件夹内的分类图片合并到一个numpy数组中, 且生成对应的labels(文件名)
       para:
           path: 数据集的路径  img_hight: 图片的高度  img_width: 图片的宽度
       return values:
           dataset: 数据集 labels: 标签
       Author: zhaozhichao
    '''
    classes = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path,item)]
    classes_num = len(classes)
    count = 0
    for item in classes:
        count += len(os.listdir(os.path.join(path, item)))
    dataset, labels = make_arrays(count, img_height, img_width)

    data_index = 0
    class_index = 0
    for item in classes:
        print("The class is: {0:}, and class index is {1:}".format(itme,str(class_index)))
        for image_name in os.listdir(os.path.join(path, item)):
            try: #read the Image
                img = np.array(Image.open(os.path.join(path, item, image_name))
                dataset[data_index,:,:] = img
                labels[data_index] = class_index
                data_index += 1
            except IOError as e:
                print(e, '- it\'s ok, skipping.')
        class_index += 1
    print("The dataset shape is:{0:}, and the labels shape is {1:}".format(str(dataset.shape), str(labels.shape))
    return dataset, labels

    # ToDo:
    # it's appear IOError, dataset and labels have empty
