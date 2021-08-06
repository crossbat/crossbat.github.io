---
layout : post
title : "[Kaggle] Intel Image Classification"
---


# [Kaggle : Intel Image Classification]
url : https://www.kaggle.com/sukeshan/image-classification-accuracy-95

# 라이브러리 불러오기


```python
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```


```python
folder_path = 'C:/Users/Windows10/Desktop/코딩/data/image_classification'
labels = os.listdir(os.path.join(folder_path, 'seg_train/seg_train'))
print(labels)
```

    ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    

# 이미지 정규화

ImageDataGenerator로는 데이터 Rescale만 해준다.


```python
train_path = os.path.join(folder_path, 'seg_train/seg_train')
Normalized = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)
data = Normalized.flow_from_directory(train_path, target_size = (150, 150), batch_size= 128, class_mode = 'sparse', shuffle = True)

```

    Found 14034 images belonging to 6 classes.
    


```python
test_path = os.path.join(folder_path, 'seg_test/seg_test')
test_data = tf.keras.preprocessing.image.ImageDataGenerator(rescale= 1/255)
test_data = test_data.flow_from_directory(test_path, target_size = (150, 150), batch_size = 128, class_mode = 'sparse', shuffle = True)

```

    Found 3000 images belonging to 6 classes.
    

# 사전훈련된 모델 불러오기(Resnet)


```python
res = tf.keras.applications.resnet_v2.ResNet50V2(
    input_shape = None,
    include_top= False, weights = 'imagenet', input_tensor= None, pooling = None,
)

# 사전에 훈련된 모델이기 때문에 다시 훈련되지 않도록 레이어를 훈련 불가능 상태로 변환
for layer in res.layers:
    layer.trainable = False

Global_pool = tf.keras.layers.GlobalAveragePooling2D()(res.output)
flat = tf.keras.layers.Flatten()(Global_pool)
drop = tf.keras.layers.Dropout(0.5)(flat)
dense1 = tf.keras.layers.Dense(1000, activation = 'relu')(drop)
drop = tf.keras.layers.Dropout(0.5)(dense1)
dense2 = tf.keras.layers.Dense(1000, activation= 'relu')(drop)
last_layer = tf.keras.layers.Dense(6, activation = 'softmax')(dense2)

model = tf.keras.Model(inputs = res.input, outputs = last_layer)

#데이터 레이블의 형태가 1차원 정수로 이루어져 있다면
# => loss = 'SparseCategorical(binary)crossentropy

model.compile(
    optimizer = tf.keras.optimizers.RMSprop(),
    loss = 'SparseCategoricalCrossentropy',
    metrics = 'accuracy'
)

model.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
    94674944/94668760 [==============================] - 12s 0us/step
    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, None, None,  0                                            
    __________________________________________________________________________________________________
    conv1_pad (ZeroPadding2D)       (None, None, None, 3 0           input_1[0][0]                    
    __________________________________________________________________________________________________
    conv1_conv (Conv2D)             (None, None, None, 6 9472        conv1_pad[0][0]                  
    __________________________________________________________________________________________________
    pool1_pad (ZeroPadding2D)       (None, None, None, 6 0           conv1_conv[0][0]                 
    __________________________________________________________________________________________________
    pool1_pool (MaxPooling2D)       (None, None, None, 6 0           pool1_pad[0][0]                  
    __________________________________________________________________________________________________
    conv2_block1_preact_bn (BatchNo (None, None, None, 6 256         pool1_pool[0][0]                 
    __________________________________________________________________________________________________
    conv2_block1_preact_relu (Activ (None, None, None, 6 0           conv2_block1_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv2_block1_1_conv (Conv2D)    (None, None, None, 6 4096        conv2_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv2_block1_1_bn (BatchNormali (None, None, None, 6 256         conv2_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_1_relu (Activation (None, None, None, 6 0           conv2_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block1_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block1_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv2_block1_2_bn (BatchNormali (None, None, None, 6 256         conv2_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_2_relu (Activation (None, None, None, 6 0           conv2_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block1_0_conv (Conv2D)    (None, None, None, 2 16640       conv2_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv2_block1_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block1_out (Add)          (None, None, None, 2 0           conv2_block1_0_conv[0][0]        
                                                                     conv2_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv2_block2_preact_relu (Activ (None, None, None, 2 0           conv2_block2_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv2_block2_1_conv (Conv2D)    (None, None, None, 6 16384       conv2_block2_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv2_block2_1_bn (BatchNormali (None, None, None, 6 256         conv2_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_1_relu (Activation (None, None, None, 6 0           conv2_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block2_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block2_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv2_block2_2_bn (BatchNormali (None, None, None, 6 256         conv2_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_2_relu (Activation (None, None, None, 6 0           conv2_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block2_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block2_out (Add)          (None, None, None, 2 0           conv2_block1_out[0][0]           
                                                                     conv2_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv2_block3_preact_relu (Activ (None, None, None, 2 0           conv2_block3_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv2_block3_1_conv (Conv2D)    (None, None, None, 6 16384       conv2_block3_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv2_block3_1_bn (BatchNormali (None, None, None, 6 256         conv2_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_1_relu (Activation (None, None, None, 6 0           conv2_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv2_block3_2_pad (ZeroPadding (None, None, None, 6 0           conv2_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_2_conv (Conv2D)    (None, None, None, 6 36864       conv2_block3_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv2_block3_2_bn (BatchNormali (None, None, None, 6 256         conv2_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_2_relu (Activation (None, None, None, 6 0           conv2_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    max_pooling2d (MaxPooling2D)    (None, None, None, 2 0           conv2_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv2_block3_3_conv (Conv2D)    (None, None, None, 2 16640       conv2_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv2_block3_out (Add)          (None, None, None, 2 0           max_pooling2d[0][0]              
                                                                     conv2_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_preact_bn (BatchNo (None, None, None, 2 1024        conv2_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block1_preact_relu (Activ (None, None, None, 2 0           conv3_block1_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv3_block1_1_conv (Conv2D)    (None, None, None, 1 32768       conv3_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv3_block1_1_bn (BatchNormali (None, None, None, 1 512         conv3_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_1_relu (Activation (None, None, None, 1 0           conv3_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block1_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block1_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv3_block1_2_bn (BatchNormali (None, None, None, 1 512         conv3_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_2_relu (Activation (None, None, None, 1 0           conv3_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block1_0_conv (Conv2D)    (None, None, None, 5 131584      conv3_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv3_block1_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block1_out (Add)          (None, None, None, 5 0           conv3_block1_0_conv[0][0]        
                                                                     conv3_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block2_preact_relu (Activ (None, None, None, 5 0           conv3_block2_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv3_block2_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block2_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv3_block2_1_bn (BatchNormali (None, None, None, 1 512         conv3_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_1_relu (Activation (None, None, None, 1 0           conv3_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block2_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block2_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv3_block2_2_bn (BatchNormali (None, None, None, 1 512         conv3_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_2_relu (Activation (None, None, None, 1 0           conv3_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block2_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block2_out (Add)          (None, None, None, 5 0           conv3_block1_out[0][0]           
                                                                     conv3_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block3_preact_relu (Activ (None, None, None, 5 0           conv3_block3_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv3_block3_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block3_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv3_block3_1_bn (BatchNormali (None, None, None, 1 512         conv3_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_1_relu (Activation (None, None, None, 1 0           conv3_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block3_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block3_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv3_block3_2_bn (BatchNormali (None, None, None, 1 512         conv3_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_2_relu (Activation (None, None, None, 1 0           conv3_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block3_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block3_out (Add)          (None, None, None, 5 0           conv3_block2_out[0][0]           
                                                                     conv3_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block4_preact_relu (Activ (None, None, None, 5 0           conv3_block4_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv3_block4_1_conv (Conv2D)    (None, None, None, 1 65536       conv3_block4_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv3_block4_1_bn (BatchNormali (None, None, None, 1 512         conv3_block4_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_1_relu (Activation (None, None, None, 1 0           conv3_block4_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv3_block4_2_pad (ZeroPadding (None, None, None, 1 0           conv3_block4_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_2_conv (Conv2D)    (None, None, None, 1 147456      conv3_block4_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv3_block4_2_bn (BatchNormali (None, None, None, 1 512         conv3_block4_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_2_relu (Activation (None, None, None, 1 0           conv3_block4_2_bn[0][0]          
    __________________________________________________________________________________________________
    max_pooling2d_1 (MaxPooling2D)  (None, None, None, 5 0           conv3_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv3_block4_3_conv (Conv2D)    (None, None, None, 5 66048       conv3_block4_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv3_block4_out (Add)          (None, None, None, 5 0           max_pooling2d_1[0][0]            
                                                                     conv3_block4_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_preact_bn (BatchNo (None, None, None, 5 2048        conv3_block4_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block1_preact_relu (Activ (None, None, None, 5 0           conv4_block1_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block1_1_conv (Conv2D)    (None, None, None, 2 131072      conv4_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block1_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_1_relu (Activation (None, None, None, 2 0           conv4_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block1_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block1_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block1_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_2_relu (Activation (None, None, None, 2 0           conv4_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block1_0_conv (Conv2D)    (None, None, None, 1 525312      conv4_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block1_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block1_out (Add)          (None, None, None, 1 0           conv4_block1_0_conv[0][0]        
                                                                     conv4_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block2_preact_relu (Activ (None, None, None, 1 0           conv4_block2_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block2_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block2_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block2_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_1_relu (Activation (None, None, None, 2 0           conv4_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block2_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block2_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block2_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_2_relu (Activation (None, None, None, 2 0           conv4_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block2_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block2_out (Add)          (None, None, None, 1 0           conv4_block1_out[0][0]           
                                                                     conv4_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block3_preact_relu (Activ (None, None, None, 1 0           conv4_block3_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block3_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block3_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block3_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_1_relu (Activation (None, None, None, 2 0           conv4_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block3_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block3_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block3_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_2_relu (Activation (None, None, None, 2 0           conv4_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block3_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block3_out (Add)          (None, None, None, 1 0           conv4_block2_out[0][0]           
                                                                     conv4_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block3_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block4_preact_relu (Activ (None, None, None, 1 0           conv4_block4_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block4_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block4_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block4_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block4_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_1_relu (Activation (None, None, None, 2 0           conv4_block4_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block4_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block4_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block4_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block4_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block4_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_2_relu (Activation (None, None, None, 2 0           conv4_block4_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block4_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block4_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block4_out (Add)          (None, None, None, 1 0           conv4_block3_out[0][0]           
                                                                     conv4_block4_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block4_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block5_preact_relu (Activ (None, None, None, 1 0           conv4_block5_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block5_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block5_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block5_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block5_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_1_relu (Activation (None, None, None, 2 0           conv4_block5_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block5_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block5_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block5_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block5_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block5_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_2_relu (Activation (None, None, None, 2 0           conv4_block5_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block5_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block5_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block5_out (Add)          (None, None, None, 1 0           conv4_block4_out[0][0]           
                                                                     conv4_block5_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block5_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block6_preact_relu (Activ (None, None, None, 1 0           conv4_block6_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv4_block6_1_conv (Conv2D)    (None, None, None, 2 262144      conv4_block6_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv4_block6_1_bn (BatchNormali (None, None, None, 2 1024        conv4_block6_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_1_relu (Activation (None, None, None, 2 0           conv4_block6_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv4_block6_2_pad (ZeroPadding (None, None, None, 2 0           conv4_block6_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_2_conv (Conv2D)    (None, None, None, 2 589824      conv4_block6_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv4_block6_2_bn (BatchNormali (None, None, None, 2 1024        conv4_block6_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_2_relu (Activation (None, None, None, 2 0           conv4_block6_2_bn[0][0]          
    __________________________________________________________________________________________________
    max_pooling2d_2 (MaxPooling2D)  (None, None, None, 1 0           conv4_block5_out[0][0]           
    __________________________________________________________________________________________________
    conv4_block6_3_conv (Conv2D)    (None, None, None, 1 263168      conv4_block6_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv4_block6_out (Add)          (None, None, None, 1 0           max_pooling2d_2[0][0]            
                                                                     conv4_block6_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_preact_bn (BatchNo (None, None, None, 1 4096        conv4_block6_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block1_preact_relu (Activ (None, None, None, 1 0           conv5_block1_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv5_block1_1_conv (Conv2D)    (None, None, None, 5 524288      conv5_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv5_block1_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block1_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_1_relu (Activation (None, None, None, 5 0           conv5_block1_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block1_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block1_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block1_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv5_block1_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block1_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_2_relu (Activation (None, None, None, 5 0           conv5_block1_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block1_0_conv (Conv2D)    (None, None, None, 2 2099200     conv5_block1_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv5_block1_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block1_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block1_out (Add)          (None, None, None, 2 0           conv5_block1_0_conv[0][0]        
                                                                     conv5_block1_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_preact_bn (BatchNo (None, None, None, 2 8192        conv5_block1_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block2_preact_relu (Activ (None, None, None, 2 0           conv5_block2_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv5_block2_1_conv (Conv2D)    (None, None, None, 5 1048576     conv5_block2_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv5_block2_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block2_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_1_relu (Activation (None, None, None, 5 0           conv5_block2_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block2_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block2_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block2_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv5_block2_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block2_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_2_relu (Activation (None, None, None, 5 0           conv5_block2_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block2_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block2_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block2_out (Add)          (None, None, None, 2 0           conv5_block1_out[0][0]           
                                                                     conv5_block2_3_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_preact_bn (BatchNo (None, None, None, 2 8192        conv5_block2_out[0][0]           
    __________________________________________________________________________________________________
    conv5_block3_preact_relu (Activ (None, None, None, 2 0           conv5_block3_preact_bn[0][0]     
    __________________________________________________________________________________________________
    conv5_block3_1_conv (Conv2D)    (None, None, None, 5 1048576     conv5_block3_preact_relu[0][0]   
    __________________________________________________________________________________________________
    conv5_block3_1_bn (BatchNormali (None, None, None, 5 2048        conv5_block3_1_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_1_relu (Activation (None, None, None, 5 0           conv5_block3_1_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block3_2_pad (ZeroPadding (None, None, None, 5 0           conv5_block3_1_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_2_conv (Conv2D)    (None, None, None, 5 2359296     conv5_block3_2_pad[0][0]         
    __________________________________________________________________________________________________
    conv5_block3_2_bn (BatchNormali (None, None, None, 5 2048        conv5_block3_2_conv[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_2_relu (Activation (None, None, None, 5 0           conv5_block3_2_bn[0][0]          
    __________________________________________________________________________________________________
    conv5_block3_3_conv (Conv2D)    (None, None, None, 2 1050624     conv5_block3_2_relu[0][0]        
    __________________________________________________________________________________________________
    conv5_block3_out (Add)          (None, None, None, 2 0           conv5_block2_out[0][0]           
                                                                     conv5_block3_3_conv[0][0]        
    __________________________________________________________________________________________________
    post_bn (BatchNormalization)    (None, None, None, 2 8192        conv5_block3_out[0][0]           
    __________________________________________________________________________________________________
    post_relu (Activation)          (None, None, None, 2 0           post_bn[0][0]                    
    __________________________________________________________________________________________________
    global_average_pooling2d (Globa (None, 2048)         0           post_relu[0][0]                  
    __________________________________________________________________________________________________
    flatten_1 (Flatten)             (None, 2048)         0           global_average_pooling2d[0][0]   
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 2048)         0           flatten_1[0][0]                  
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 1000)         2049000     dropout[0][0]                    
    __________________________________________________________________________________________________
    dropout_1 (Dropout)             (None, 1000)         0           dense_2[0][0]                    
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 1000)         1001000     dropout_1[0][0]                  
    __________________________________________________________________________________________________
    dense_4 (Dense)                 (None, 6)            6006        dense_3[0][0]                    
    ==================================================================================================
    Total params: 26,620,806
    Trainable params: 3,056,006
    Non-trainable params: 23,564,800
    __________________________________________________________________________________________________
    

# Callback 함수 지정


```python
# EarlyStopping은 정해진 조건에 맞춰서 오버피팅되는 순간에 훈련을 멈춰줌

early = tf.keras.callbacks.EarlyStopping(monitor= 'val_loss', patience= 5)
callback_list = [early]
```


```python
history = model.fit(data, validation_data = test_data, epochs =1)
```

    110/110 [==============================] - 603s 5s/step - loss: 0.3755 - accuracy: 0.8649 - val_loss: 0.3795 - val_accuracy: 0.8410
    


```python
model.evaluate(data)
```

    110/110 [==============================] - 466s 4s/step - loss: 0.3587 - accuracy: 0.8524
    




    [0.35871434211730957, 0.8523585796356201]




```python

```
