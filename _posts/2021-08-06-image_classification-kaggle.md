---
layout : post
title : 'Kaggle Intel Image Classification - 2'
---

# [Kaggle Intel Image classification]

    앞에 올린 것은 나의 생각이 전혀 반영되지 않은 방식이기에

    어느정도 내가 의도한 방법과 도움을 조금 받아서 다시 작성하였다.

# - 라이브러리 불러오기


```python
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
```

# - 데이터 위치 지정


```python
data_dir = 'C:/Users/Windows10/Desktop/코딩/data/image_classification'
train_dir = os.path.join(data_dir, 'seg_train/seg_train')
val_dir = os.path.join(data_dir, 'seg_test/seg_test')

pred_dir = os.path.join(data_dir, 'seg_pred/seg_pred')
```


```python
#데이터 분류 성분

classes = os.listdir(train_dir)
print(classes)
```

    ['buildings', 'forest', 'sea', 'mountain', 'glacier', 'street']
    

# - 훈련할 데이터 살펴보기


```python
num_data = {}
for place in classes:
    num_data[place] = len(glob(os.path.join(train_dir, place + '/*')))

print(num_data)
```

    {'buildings': 2191, 'forest': 2271, 'sea': 2274, 'mountain': 2512, 'glacier': 2404, 'street': 2382}
    


```python
plt.figure(figsize = (8,6))
plt.pie(num_data.values(), labels = classes, autopct='%.1f')
plt.show()
```


    
![output_8_0](https://user-images.githubusercontent.com/86095931/128486985-594fd81c-fd79-449b-898c-e9bb4c12a5ca.png)

    


# - 데이터 전처리


```python
batch_size = 32
input_shape = (180, 180)
```


```python
train_datagen = ImageDataGenerator(rescale= 1./255, horizontal_flip= True, zoom_range = 0.4, shear_range= 0.3)
test_datagen = ImageDataGenerator(rescale=1./255)
```

* 한 데이터를 오로지 훈련셋으로 사용하려면 validation_split을 지정하지 않아도 된다.


```python
train_ds = train_datagen.flow_from_directory(train_dir, shuffle = True, target_size = input_shape, batch_size = batch_size, class_mode = 'sparse')
print(len(train_ds))
```

    Found 14034 images belonging to 6 classes.
    439
    


```python
val_ds = test_datagen.flow_from_directory(val_dir, shuffle= True, target_size = input_shape, batch_size = batch_size, class_mode = 'sparse')
```

    Found 3000 images belonging to 6 classes.
    


```python
prediction_imgs = list(glob(os.path.join(pred_dir, '*')))

len(prediction_imgs)
```




    7301



- 테스트 할 데이터 셋 전처리


```python
def get_data(path):
    x = []
    for i in path:
        img = load_img(i, target_size = (180, 180), color_mode = 'rgb')
        img = img_to_array(img)
        img /= 255
        x.append(img)
    x = np.asarray(x)
    return x
```


```python
test_ds = get_data(prediction_imgs)
```


```python
print(test_ds.shape)
```




    (7301, 180, 180, 3)




```python
# 미리보기

plt.figure(figsize = (10, 10))
for i in range(20):
    plt.subplot(5, 4, i + 1)
    plt.imshow(test_ds[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
```


    
![output_20_0](https://user-images.githubusercontent.com/86095931/128487072-56f4535d-cc51-4298-850a-b8c8c5b06b14.png)

    



```python
input_shape = input_shape + (3,)

#모델에 들어갈 input_shape는 (img_height, img_width, filters)이기 때문에
#컬러 데이터는 filters가 3이고,
#흑백 데이터의 경우, 1이다.
```


```python
model = Sequential([
    Conv2D(64, 4, activation = 'relu', input_shape = input_shape),
    MaxPool2D(pool_size = (2,2)),
    Conv2D(128, 4, activation = 'relu'),
    MaxPool2D(pool_size = (2,2)),
    Conv2D(64, 4, activation = 'relu'),
    MaxPool2D(pool_size = (2,2)),
    Conv2D(64, 4, activation = 'relu'),
    MaxPool2D(pool_size = (2,2)),
    Dropout(0.3),       #Dropout : 모델의 일부분을 전원을 꺼두듯이 사용하지 않게끔 하는 것
    Flatten(),
    Dense(128, activation = 'relu'),
    Dropout(0.4),
    Dense(6, activation = 'softmax')
])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 177, 177, 64)      3136      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 88, 88, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 85, 85, 128)       131200    
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 42, 42, 128)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 39, 39, 64)        131136    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 19, 19, 64)        0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 16, 16, 64)        65600     
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 8, 8, 64)          0         
    _________________________________________________________________
    dropout (Dropout)            (None, 8, 8, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 4096)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               524416    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 6)                 774       
    =================================================================
    Total params: 856,262
    Trainable params: 856,262
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(optimizer = 'adam', loss=  'sparse_categorical_crossentropy', metrics= ['acc'])

#데이터의 레이블이 1차원 정수로 이루어져 있을 때는
#위와같이 sparse_categorical(binary)_crossentropy를,
#2차원이상이거나 one-hot encoding으로 되어있을 경우에는
#categorical(binary)_crossentropy를 사용한다.
```

- 모델을 학습시킬 때, fit_generator를 사용하게 된다면
- steps_per_epoch라는 것을 써야하는데,
- 일반적으로 (학습시킬 데이터의 수 / 배치사이즈) 인데
- 배치사이즈로 나누지 않는 경우가 더 많은 것 같다.


```python
history = model.fit(train_ds, steps_per_epoch= len(train_ds), epochs = 10, validation_data = val_ds)
```

    Epoch 1/10
    439/439 [==============================] - 178s 336ms/step - loss: 1.1949 - acc: 0.5151 - val_loss: 0.9411 - val_acc: 0.6093
    Epoch 2/10
    439/439 [==============================] - 145s 330ms/step - loss: 0.9636 - acc: 0.6216 - val_loss: 0.8258 - val_acc: 0.6880
    Epoch 3/10
    439/439 [==============================] - 146s 332ms/step - loss: 0.8281 - acc: 0.6905 - val_loss: 0.6647 - val_acc: 0.7547
    Epoch 4/10
    439/439 [==============================] - 145s 330ms/step - loss: 0.7338 - acc: 0.7356 - val_loss: 0.6333 - val_acc: 0.7600
    Epoch 5/10
    439/439 [==============================] - 146s 332ms/step - loss: 0.6622 - acc: 0.7617 - val_loss: 0.5227 - val_acc: 0.8100
    Epoch 6/10
    439/439 [==============================] - 146s 332ms/step - loss: 0.6190 - acc: 0.7812 - val_loss: 0.5503 - val_acc: 0.7983
    Epoch 7/10
    439/439 [==============================] - 145s 331ms/step - loss: 0.5912 - acc: 0.7906 - val_loss: 0.4676 - val_acc: 0.8290
    Epoch 8/10
    439/439 [==============================] - 146s 332ms/step - loss: 0.5697 - acc: 0.8028 - val_loss: 0.4781 - val_acc: 0.8253
    Epoch 9/10
    439/439 [==============================] - 145s 330ms/step - loss: 0.5275 - acc: 0.8132 - val_loss: 0.4173 - val_acc: 0.8570
    Epoch 10/10
    439/439 [==============================] - 145s 330ms/step - loss: 0.5202 - acc: 0.8167 - val_loss: 0.4321 - val_acc: 0.8423
    


```python
val_loss, val_acc = model.evaluate(val_ds)
```

    94/94 [==============================] - 7s 73ms/step - loss: 0.4321 - acc: 0.8423
    


```python
plt.figure(figsize = (8,6))
plt.subplot(1,2,1)
plt.plot(history.history['acc'], label = 'training')
plt.plot(history.history['val_acc'], label = 'validation')
plt.xlabel('Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label = 'training')
plt.plot(history.history['val_loss'], label = 'validation')
plt.xlabel('Loss')
plt.legend()
plt.show()
```


    
![output_27_0](https://user-images.githubusercontent.com/86095931/128487142-f69a348f-58c9-48db-a37f-5915de49c74d.png)

    



```python
test_ds[0].shape
```




    (180, 180, 3)




```python
predict = model.predict(test_ds)
```


```python
predict = tf.nn.softmax(predict)
```


```python
#predict의 내용을 가장 큰 확률의 값만 나오게 함
predict = np.argmax(predict, axis = 1)
```


```python
labels = list(num_data.keys())
labels
```




    ['buildings', 'forest', 'sea', 'mountain', 'glacier', 'street']




```python
import random

for i in range(0, 9):
  plt.subplot(3,3, i + 1)
  r = random.randint(0, len(test_ds))
  plt.imshow(test_ds[r])
  plt.xlabel(labels[predict[r]])
  plt.xticks([])
  plt.yticks([])

plt.show()
```


    
![output_33_0](https://user-images.githubusercontent.com/86095931/128487189-c22ca381-6ed6-4144-8af1-95a6c52fd97c.png)

    

