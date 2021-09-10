---
layout = 'post'
title = X-ray Classification
---

```python
import os
from glob import glob
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
```


```python
zip_dir = '/content/drive/MyDrive/Data/archive (1).zip'

zip_data = zipfile.ZipFile(zip_dir)

zip_data.extractall('/content')
```


```python
data_dir = '/content/chest_xray'
```


```python
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')
```


```python
for i in range(9):
    plt.subplot(3,3,i + 1)
    if (i + 1) %2 == 0:
        image = img.imread(glob(os.path.join(train_dir, 'NORMAL/*'))[i])
        label = 'normal'
    else:
        image = img.imread(glob(os.path.join(train_dir, 'PNEUMONIA/*'))[i])
        label = 'pneumonia'

    plt.imshow(image)
    plt.xlabel(label)

plt.show()

```


    
![output_4_0](https://user-images.githubusercontent.com/86095931/132815100-3cc7ac46-36f4-45e3-a1f2-93a769829cf8.png)
    



```python
train_gen = ImageDataGenerator(
    rescale = 1/255,
    width_shift_range= 0.2,
    height_shift_range=0.2,
    shear_range= 0.2,
    zoom_range= 0.2,
    horizontal_flip= True,
)

test_gen = ImageDataGenerator(
    rescale = 1/255,
)

val_gen = ImageDataGenerator(
    rescale = 1/255
)
```


```python
train_set = train_gen.flow_from_directory(
    train_dir,
    target_size = (180, 180),
    class_mode = 'binary',
    batch_size = 64,
)

test_set = test_gen.flow_from_directory(
    test_dir,
    target_size = (180, 180),
    class_mode = 'binary',
    batch_size = 64,
)

val_set = val_gen.flow_from_directory(
    val_dir,
    target_size = (180, 180),
    batch_size = 64,
    class_mode = 'binary'
)
```

    Found 5216 images belonging to 2 classes.
    Found 624 images belonging to 2 classes.
    Found 16 images belonging to 2 classes.
    


```python
model = Sequential([
    Conv2D(64, 3, padding = 'same', activation= 'relu', input_shape = (180, 180, 3)),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(128, 3, padding= 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(128, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(256, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(512, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(512, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Conv2D(1024, 3, padding = 'same', activation = 'relu'),
    MaxPooling2D(pool_size = (2,2)),
    Dropout(0.3),
    Flatten(),
    Dense(1024, activation = 'relu'),
    Dense(256, activation = 'relu'),
    Dense(64, activation = 'relu'),
    Dense(1, activation = 'sigmoid'),
])

model.summary()

import pydot
import graphviz
from tensorflow.keras.utils import plot_model

plot_model(model)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 180, 180, 64)      1792      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 90, 90, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 90, 90, 128)       73856     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 45, 45, 128)       0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 45, 45, 128)       147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 22, 22, 128)       0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 22, 22, 256)       295168    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 11, 11, 256)       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 11, 11, 512)       1180160   
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 5, 5, 512)         0         
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 5, 5, 512)         2359808   
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 2, 2, 512)         0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 2, 2, 1024)        4719616   
    _________________________________________________________________
    max_pooling2d_6 (MaxPooling2 (None, 1, 1, 1024)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 1, 1, 1024)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1024)              0         
    _________________________________________________________________
    dense (Dense)                (None, 1024)              1049600   
    _________________________________________________________________
    dense_1 (Dense)              (None, 256)               262400    
    _________________________________________________________________
    dense_2 (Dense)              (None, 64)                16448     
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 65        
    =================================================================
    Total params: 10,106,497
    Trainable params: 10,106,497
    Non-trainable params: 0
    _________________________________________________________________
    




    
![output_7_1](https://user-images.githubusercontent.com/86095931/132815421-de7e920f-159d-46c6-8bb2-df6a59d4054b.png)

    




```python
model.compile(loss = 'binary_crossentropy', optimizer= 'rmsprop', metrics = ['acc'])
```


```python
es = EarlyStopping(monitor= 'val_loss', patience= 5, verbose= 1)
mc = ModelCheckpoint(filepath= os.path.join(data_dir, 'log'), monitor= 'val_loss', save_weights_only= True, save_freq= 'epoch')
```


```python
callbacks = [es, mc]
```


```python
history = model.fit_generator(train_set, steps_per_epoch= 50, epochs = 50, callbacks = callbacks, validation_data = test_set, validation_steps = 5)
```

    /usr/local/lib/python3.7/dist-packages/keras/engine/training.py:1972: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      warnings.warn('`Model.fit_generator` is deprecated and '
    

    Epoch 1/50
    50/50 [==============================] - 100s 1s/step - loss: 0.9449 - acc: 0.7238 - val_loss: 0.8376 - val_acc: 0.6000
    Epoch 2/50
    50/50 [==============================] - 63s 1s/step - loss: 0.6065 - acc: 0.7399 - val_loss: 0.6747 - val_acc: 0.6062
    Epoch 3/50
    50/50 [==============================] - 64s 1s/step - loss: 0.5100 - acc: 0.7421 - val_loss: 0.5592 - val_acc: 0.6219
    Epoch 4/50
    50/50 [==============================] - 64s 1s/step - loss: 0.4503 - acc: 0.7622 - val_loss: 0.7312 - val_acc: 0.4719
    Epoch 5/50
    50/50 [==============================] - 64s 1s/step - loss: 0.4550 - acc: 0.7891 - val_loss: 0.6138 - val_acc: 0.6438
    Epoch 6/50
    50/50 [==============================] - 63s 1s/step - loss: 0.3909 - acc: 0.8269 - val_loss: 0.8792 - val_acc: 0.7625
    Epoch 7/50
    50/50 [==============================] - 63s 1s/step - loss: 0.3640 - acc: 0.8428 - val_loss: 0.5262 - val_acc: 0.7656
    Epoch 8/50
    50/50 [==============================] - 63s 1s/step - loss: 0.3689 - acc: 0.8384 - val_loss: 0.5687 - val_acc: 0.6313
    Epoch 9/50
    50/50 [==============================] - 64s 1s/step - loss: 0.3318 - acc: 0.8617 - val_loss: 0.4722 - val_acc: 0.7625
    Epoch 10/50
    50/50 [==============================] - 63s 1s/step - loss: 0.3045 - acc: 0.8699 - val_loss: 0.5304 - val_acc: 0.7812
    Epoch 11/50
    50/50 [==============================] - 62s 1s/step - loss: 0.3331 - acc: 0.8718 - val_loss: 0.3675 - val_acc: 0.8562
    Epoch 12/50
    50/50 [==============================] - 63s 1s/step - loss: 0.2858 - acc: 0.8826 - val_loss: 0.4017 - val_acc: 0.8375
    Epoch 13/50
    50/50 [==============================] - 66s 1s/step - loss: 0.2828 - acc: 0.8894 - val_loss: 0.4577 - val_acc: 0.7688
    Epoch 14/50
    50/50 [==============================] - 65s 1s/step - loss: 0.2600 - acc: 0.8971 - val_loss: 0.3688 - val_acc: 0.8687
    Epoch 15/50
    50/50 [==============================] - 65s 1s/step - loss: 0.3106 - acc: 0.8731 - val_loss: 0.3485 - val_acc: 0.8500
    Epoch 16/50
    50/50 [==============================] - 65s 1s/step - loss: 0.2618 - acc: 0.8924 - val_loss: 0.6318 - val_acc: 0.7969
    Epoch 17/50
    50/50 [==============================] - 65s 1s/step - loss: 0.2566 - acc: 0.9038 - val_loss: 0.3036 - val_acc: 0.9031
    Epoch 18/50
    50/50 [==============================] - 65s 1s/step - loss: 0.2333 - acc: 0.9081 - val_loss: 0.4802 - val_acc: 0.8781
    Epoch 19/50
    50/50 [==============================] - 64s 1s/step - loss: 0.2494 - acc: 0.9025 - val_loss: 0.3959 - val_acc: 0.8906
    Epoch 20/50
    50/50 [==============================] - 64s 1s/step - loss: 0.2188 - acc: 0.9153 - val_loss: 0.9143 - val_acc: 0.7469
    Epoch 21/50
    50/50 [==============================] - 65s 1s/step - loss: 0.2315 - acc: 0.9097 - val_loss: 0.3912 - val_acc: 0.8250
    Epoch 22/50
    50/50 [==============================] - 66s 1s/step - loss: 0.2141 - acc: 0.9206 - val_loss: 1.5678 - val_acc: 0.7156
    Epoch 00022: early stopping
    


```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoch = range(len(acc))

plt.figure(figsize = (12, 12)),
plt.plot(epoch, acc, 'r-', label = 'training acc')
plt.plot(epoch, val_acc, 'b-', label = 'validation acc')
plt.title('training and validation acc')
plt.legend()

plt.figure(figsize=(12, 12))
plt.plot(epoch, loss, 'r-', label = 'training loss')
plt.plot(epoch, val_loss, 'b-', label = 'validation loss')
plt.title('training and validation loss')
plt.legend()

plt.show()
```


    
![output_12_0](https://user-images.githubusercontent.com/86095931/132815465-e3966867-2082-4e52-a8ed-fcd753fb3d72.png)
    



    
![output_12_1](https://user-images.githubusercontent.com/86095931/132815493-a66b3de1-9ec8-4aa9-afb5-fd82c8be8dea.png)
    



```python

```
