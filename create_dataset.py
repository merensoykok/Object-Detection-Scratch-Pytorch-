import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import shuffle
import xml.etree.ElementTree as ET
import time
from sklearn.utils import shuffle

from PIL import Image
import torchvision.transforms as transforms


def encode_class(class_name) -> list:
    class_label_by_name = {"Basketball": 0,
                           "Football": 1,
                           "Volleyball": 2
                           }
    return class_label_by_name[class_name]


def parse_xml(xml_path: str) -> list:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    filename = root.find('filename').text

    file_names = []
    list_ymin = []
    list_ymax = []
    list_xmin = []
    list_xmax = []
    list_class_name = []

    for boxes in root.iter('object'):
        class_name = boxes.find("name").text
        class_labeled = encode_class(class_name)

        ymin = int(boxes.find("bndbox/ymin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        file_names.append(filename)
        list_class_name.append(class_labeled)
        list_ymin.append(ymin)
        list_ymax.append(ymax)
        list_xmin.append(xmin)
        list_xmax.append(xmax)

    return file_names, list_class_name, list_ymin, list_ymax, list_xmin, list_xmax


def create_train_val_test_subsets(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    grouped = df.groupby("class")

    class_counts = grouped.size().values

    # Her sınıftan train, validation ve test setlerine eklemek için örnek sayılarını hesaplayın
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    train_samples = (class_counts * train_ratio).astype(int)
    val_samples = (class_counts * val_ratio).astype(int)
    test_samples = (class_counts * test_ratio).astype(int)

    train_set = pd.concat([group.sample(train_samples[i]) for i, (_, group) in enumerate(grouped)])
    validation_set = pd.concat([group.sample(val_samples[i]) for i, (_, group) in enumerate(grouped)])
    test_set = pd.concat([group.sample(test_samples[i]) for i, (_, group) in enumerate(grouped)])
   
    train_set = train_set.sample(frac=1, random_state=42).reset_index(drop=True)
    validation_set = validation_set.sample(frac=1, random_state=42).reset_index(drop=True)
    test_set = test_set.sample(frac=1, random_state=42).reset_index(drop=True)

    """
    np.random.seed(42)
    np.random.shuffle(train_set.values)
    np.random.seed(42)
    np.random.shuffle(validation_set.values)
    np.random.seed(42)
    np.random.shuffle(test_set.values)
    
    train_set.reset_index(inplace=True)
    validation_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)
    """
    
    return train_set, validation_set, test_set


def create_dataset_df(dataset_path: str) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    file_list = os.listdir(dataset_path)

    files = []
    classes = []
    ymins = []
    ymaxs = []
    xmins = []
    xmaxs = []

    for i in range(0, len(file_list), 2):
        xml = file_list[i + 1]

        xml_path = os.path.join(dataset_path, xml)
        file_names, list_class_name, list_ymin, list_ymax, list_xmin, list_xmax = parse_xml(xml_path)

        files.extend(file_names)
        classes.extend(list_class_name)
        ymins.extend(list_ymin)
        ymaxs.extend(list_ymax)
        xmins.extend(list_xmin)
        xmaxs.extend(list_xmax)

    df = pd.DataFrame({
        "file_names": files,
        "class": classes,
        "ymin": ymins,
        "ymax": ymaxs,
        "xmin": xmins,
        "xmax": xmaxs
    })

    train_set, validation_set, test_set = create_train_val_test_subsets(df)

    return train_set, validation_set, test_set


def main():
    start_time = time.perf_counter()

    dataset_path = r"C:\Users\Eren\Downloads\Ball-Detection.v1i.voc\train"
    dataset_type = ["train", "val", "test"]

    os.chdir(dataset_path)

    train_set, validation_set, test_set = create_dataset_df(dataset_path)



    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("Elapsed time: ", elapsed_time)


if __name__ == '__main__':
    main()