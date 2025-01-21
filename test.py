import os
import cv2
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from arrange_data import IMG_SIZE
from model import model
from tensorflow.keras.utils import to_categorical

# Test verisini yükleme fonksiyonu
def load_test_data(test_dir, img_size):
    X_test = []
    y_test = []
    for category in ['closed', 'open']: 
        path = os.path.join(test_dir, category)
        class_label = 0 if category == 'open' else 1 
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
                img = cv2.resize(img, (img_size, img_size))  
                X_test.append(img)
                y_test.append(class_label)
            except Exception as e:
                print(f"Bir hata oluştu: {e} - {img_name}")
    X_test = np.array(X_test).reshape(-1, img_size, img_size, 1) / 255.0 
    y_test = np.array(y_test)
    return X_test, y_test

# Test verisini yükle
test_dir = r"C:\Users\HP\Desktop\CS48007-Project\dataset\test"
X_test, y_test = load_test_data(test_dir, IMG_SIZE)

y_test_categorical = to_categorical(y_test, num_classes=2)

# Modelin tahminleri
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test_categorical, axis=1)


# Metikleri hesapla
accuracy = accuracy_score(y_true_classes, y_pred_classes)
precision = precision_score(y_true_classes, y_pred_classes, average='weighted')
recall = recall_score(y_true_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_true_classes, y_pred_classes, average='weighted')

# Sonuçları yazdır
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Daha ayrıntılı rapor
report = classification_report(y_true_classes, y_pred_classes, target_names=['Kapalı', 'Açık'])
print("\nClassification Report:\n", report)

import matplotlib.pyplot as plt
from model import X_train

# Eğitim ve test veri setinden rastgele görüntüler görselleştir
plt.imshow(X_train[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
plt.title('Eğitim Verisi Örneği')
plt.show()

plt.imshow(X_test[0].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
plt.title('Test Verisi Örneği')
plt.show()
