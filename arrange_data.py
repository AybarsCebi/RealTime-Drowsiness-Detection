import os
import cv2
import numpy as np

# Train Veri yolları
data_dir = r"C:\Users\HP\Desktop\CS48007-Project\dataset\train"
categories = ['open', 'closed']


IMG_SIZE = 64

def load_data(data_dir, categories):
    data = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)  # open -> 0, closed -> 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([img_array, label])
            except Exception as e:
                print(f"Hata: {e}")
    return data


data = load_data(data_dir, categories)
import random
random.shuffle(data)

X = np.array([item[0] for item in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalizasyon
y = np.array([item[1] for item in data])
print(f"Girdi veri boyutu: {X.shape}")
print(f"Etiket veri boyutu: {y.shape}")




# Test veri yolları
testdata_dir = r"C:\Users\HP\Desktop\CS48007-Project\dataset\test"
test_categories = ['open', 'closed']


IMG_SIZE = 64

def load_data(testdata_dir, test_categories):
    data = []
    for category in test_categories:
        path = os.path.join(testdata_dir, category)
        label = test_categories.index(category)  # open -> 0, closed -> 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([img_array, label])
            except Exception as e:
                print(f"Hata: {e}")
    return data


data = load_data(testdata_dir, test_categories)
random.shuffle(data)

testX = np.array([item[0] for item in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  
testy = np.array([item[1] for item in data])
print(f"Girdi veri boyutu: {testX.shape}")
print(f"Etiket veri boyutu: {testy.shape}")
