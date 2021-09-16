---
layout : post
title : '주가 예측해보기'
---

# 삼성주가 예측 해보기(테디노트 참고)

혼자 하는데 학습도 제대로 안되고 해서 참고해서 했다. 어떤 부분이 잘못됬는지 열심히 찾아봤는데 못찾았다...

- 라이브러리 


```python
import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
```

- 삼성주가 데이터 불러오기


```python
stock = fdr.DataReader('005930')
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-08-29</th>
      <td>1265</td>
      <td>1295</td>
      <td>1256</td>
      <td>1265</td>
      <td>149530</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1997-08-30</th>
      <td>1259</td>
      <td>1259</td>
      <td>1236</td>
      <td>1260</td>
      <td>128610</td>
      <td>-0.003953</td>
    </tr>
    <tr>
      <th>1997-09-01</th>
      <td>1258</td>
      <td>1268</td>
      <td>1236</td>
      <td>1251</td>
      <td>76170</td>
      <td>-0.007143</td>
    </tr>
    <tr>
      <th>1997-09-02</th>
      <td>1238</td>
      <td>1268</td>
      <td>1227</td>
      <td>1269</td>
      <td>97370</td>
      <td>0.014388</td>
    </tr>
    <tr>
      <th>1997-09-03</th>
      <td>1268</td>
      <td>1268</td>
      <td>1236</td>
      <td>1237</td>
      <td>108600</td>
      <td>-0.025217</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2021-09-09</th>
      <td>76400</td>
      <td>76600</td>
      <td>75000</td>
      <td>75300</td>
      <td>17600770</td>
      <td>-0.013106</td>
    </tr>
    <tr>
      <th>2021-09-10</th>
      <td>75300</td>
      <td>75600</td>
      <td>74800</td>
      <td>75300</td>
      <td>10103212</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2021-09-13</th>
      <td>75200</td>
      <td>76300</td>
      <td>75100</td>
      <td>76300</td>
      <td>11397775</td>
      <td>0.013280</td>
    </tr>
    <tr>
      <th>2021-09-14</th>
      <td>77100</td>
      <td>77700</td>
      <td>76600</td>
      <td>76600</td>
      <td>18167057</td>
      <td>0.003932</td>
    </tr>
    <tr>
      <th>2021-09-15</th>
      <td>77400</td>
      <td>77400</td>
      <td>76600</td>
      <td>76800</td>
      <td>5030953</td>
      <td>0.002611</td>
    </tr>
  </tbody>
</table>
</div>




```python
stock.index
```




    DatetimeIndex(['1997-08-29', '1997-08-30', '1997-09-01', '1997-09-02',
                   '1997-09-03', '1997-09-04', '1997-09-05', '1997-09-06',
                   '1997-09-08', '1997-09-09',
                   ...
                   '2021-09-02', '2021-09-03', '2021-09-06', '2021-09-07',
                   '2021-09-08', '2021-09-09', '2021-09-10', '2021-09-13',
                   '2021-09-14', '2021-09-15'],
                  dtype='datetime64[ns]', name='Date', length=6000, freq=None)




```python
stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day
```


```python
stock.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Close</th>
      <th>Volume</th>
      <th>Change</th>
      <th>Year</th>
      <th>Month</th>
      <th>Day</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1997-08-29</th>
      <td>1265</td>
      <td>1295</td>
      <td>1256</td>
      <td>1265</td>
      <td>149530</td>
      <td>NaN</td>
      <td>1997</td>
      <td>8</td>
      <td>29</td>
    </tr>
    <tr>
      <th>1997-08-30</th>
      <td>1259</td>
      <td>1259</td>
      <td>1236</td>
      <td>1260</td>
      <td>128610</td>
      <td>-0.003953</td>
      <td>1997</td>
      <td>8</td>
      <td>30</td>
    </tr>
    <tr>
      <th>1997-09-01</th>
      <td>1258</td>
      <td>1268</td>
      <td>1236</td>
      <td>1251</td>
      <td>76170</td>
      <td>-0.007143</td>
      <td>1997</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1997-09-02</th>
      <td>1238</td>
      <td>1268</td>
      <td>1227</td>
      <td>1269</td>
      <td>97370</td>
      <td>0.014388</td>
      <td>1997</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1997-09-03</th>
      <td>1268</td>
      <td>1268</td>
      <td>1236</td>
      <td>1237</td>
      <td>108600</td>
      <td>-0.025217</td>
      <td>1997</td>
      <td>9</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>


- 데이터 시각화

```python
plt.figure(figsize = (16, 9))
sns.lineplot(y = stock['Close'], x = stock.index)
plt.xlabel('time')
plt.ylabel('price')
```




    Text(0, 0.5, 'price')




    
![output_6_1](https://user-images.githubusercontent.com/86095931/133551277-068da423-3720-49ba-b977-7d0651a04192.png)
    


- 데이터 년도별로 나누어서 보기
*근데 뭔가 좀 이상하게 됨

```python
time_steps = [['1990', '2000'],
              ['2000', '2010'],
              ['2010', '2015'],
              ['2015', '2020']]

fig, axes = plt.subplots(2,2)
fig.set_size_inches(16, 9)
for i in range(4):
  ax = axes[i //2, 1%2]
  df = stock.loc[(stock.index > time_steps[i][0]) & (stock.index < time_steps[i][1])]
  sns.lineplot(y = df['Close'], x = df.index, ax = ax)
  ax.set_title(f'{time_steps[i][0]} ~ {time_steps[i][1]}')
  ax.set_xlabel('time')
  ax.set_ylabel('price')
plt.tight_layout()
plt.show()
```


    
![output_7_0](https://user-images.githubusercontent.com/86095931/133551417-054ac8ee-de52-4d6e-a121-2e3386e91d53.png)
    

- 데이터 전처리

```python
scaler = MinMaxScaler()

scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

scaled = scaler.fit_transform(stock[scale_cols])
scaled
```




    array([[1.40088594e-02, 1.33780992e-02, 1.40335196e-02, 7.05963064e-03,
            1.65581143e-03],
           [1.39424142e-02, 1.30061983e-02, 1.38100559e-02, 7.00430438e-03,
            1.42415507e-03],
           [1.39313400e-02, 1.30991736e-02, 1.38100559e-02, 6.90471712e-03,
            8.43463897e-04],
           ...,
           [8.32779623e-01, 7.88223140e-01, 8.39106145e-01, 8.37340799e-01,
            1.26212573e-01],
           [8.53820598e-01, 8.02685950e-01, 8.55865922e-01, 8.40660374e-01,
            2.01171809e-01],
           [8.57142857e-01, 7.99586777e-01, 8.55865922e-01, 8.42873425e-01,
            5.57099544e-02]])




```python
df = pd.DataFrame(scaled, columns = scale_cols)
```

- 데이터 분할
```python
x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', axis = 1), df['Close'], test_size = 0.2, shuffle = False)
```


```python
x_train.shape, y_train.shape
```




    ((4800, 4), (4800,))




```python
x_test.shape, y_test.shape
```




    ((1200, 4), (1200,))




```python
x_train
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Open</th>
      <th>High</th>
      <th>Low</th>
      <th>Volume</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.014009</td>
      <td>0.013378</td>
      <td>0.014034</td>
      <td>0.001656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.013942</td>
      <td>0.013006</td>
      <td>0.013810</td>
      <td>0.001424</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.013931</td>
      <td>0.013099</td>
      <td>0.013810</td>
      <td>0.000843</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.013710</td>
      <td>0.013099</td>
      <td>0.013709</td>
      <td>0.001078</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.014042</td>
      <td>0.013099</td>
      <td>0.013810</td>
      <td>0.001203</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4795</th>
      <td>0.353699</td>
      <td>0.330372</td>
      <td>0.349050</td>
      <td>0.002333</td>
    </tr>
    <tr>
      <th>4796</th>
      <td>0.347951</td>
      <td>0.334091</td>
      <td>0.347709</td>
      <td>0.003126</td>
    </tr>
    <tr>
      <th>4797</th>
      <td>0.349945</td>
      <td>0.333471</td>
      <td>0.353073</td>
      <td>0.002262</td>
    </tr>
    <tr>
      <th>4798</th>
      <td>0.357918</td>
      <td>0.338636</td>
      <td>0.360000</td>
      <td>0.002673</td>
    </tr>
    <tr>
      <th>4799</th>
      <td>0.361019</td>
      <td>0.341322</td>
      <td>0.360223</td>
      <td>0.002292</td>
    </tr>
  </tbody>
</table>
<p>4800 rows × 4 columns</p>
</div>



- 데이터셋 만드는 함수 지정

이해는 되는데 잘 못써먹겠다. 많이 써야지

```python
def windowed_dataset(series, window_size, batch_size, shuffle):
  series = tf.expand_dims(series, axis = -1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift = 1, drop_remainder =True)
  ds = ds.flat_map(lambda w : w.batch(window_size + 1))
  if shuffle:
    ds = ds.shuffle(1000)
  ds = ds.map(lambda w : (w[:-1], w[-1]))
  return ds.batch(batch_size).prefetch(1)
```


```python
window_size = 20
batch_size = 32
```


```python
train_data = windowed_dataset(y_train, window_size, batch_size, True)
test_data = windowed_dataset(y_test, window_size, batch_size, False)
```


```python
for data in train_data.take(1):
  print(data[0].shape)
  print(data[1].shape)
```

    (32, 20, 1)
    (32, 1)
    
- 모델 만들기

```python
model = Sequential([
    Conv1D(filters = 32, kernel_size = 5, padding = 'causal', activation = 'relu', input_shape = [window_size, 1]),
    LSTM(16, activation = 'tanh'),
    Dense(16, activation  = 'relu'),
    Dense(1),
])
```

- 모델 컴파일
```python
loss = Huber()
optimizer = Adam(0.005)
model.compile(loss = loss, optimizer = optimizer\, metrics = ['mse'])
```

- 콜백 함수 지정
```python
es = EarlyStopping(monitor = 'val_loss', patience = 10)
mc = ModelCheckpoint('/content/checkopint_logs',
                     save_best_only = True,
                     save_weights_only = True,
                     monitor = 'val_loss',
                     verbose = 1
                     )
```

- 모델 학습
```python
history = model.fit(train_data, validation_data = (test_data),
                    epochs = 50, callbacks = [es, mc])
```

    Epoch 1/50
    150/150 [==============================] - 5s 19ms/step - loss: 3.6151e-05 - mse: 7.2301e-05 - val_loss: 0.0023 - val_mse: 0.0046
    
    Epoch 00001: val_loss improved from inf to 0.00230, saving model to /content/checkopint_logs
    Epoch 2/50
    150/150 [==============================] - 2s 15ms/step - loss: 2.1790e-05 - mse: 4.3580e-05 - val_loss: 0.0028 - val_mse: 0.0056
    
    Epoch 00002: val_loss did not improve from 0.00230
    Epoch 3/50
    150/150 [==============================] - 2s 15ms/step - loss: 2.3798e-05 - mse: 4.7596e-05 - val_loss: 0.0037 - val_mse: 0.0073
    
    Epoch 00003: val_loss did not improve from 0.00230
    Epoch 4/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.6514e-05 - mse: 3.3029e-05 - val_loss: 0.0019 - val_mse: 0.0038
    
    Epoch 00004: val_loss improved from 0.00230 to 0.00189, saving model to /content/checkopint_logs
    Epoch 5/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.3408e-05 - mse: 2.6816e-05 - val_loss: 0.0010 - val_mse: 0.0021
    
    Epoch 00005: val_loss improved from 0.00189 to 0.00104, saving model to /content/checkopint_logs
    Epoch 6/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.4419e-05 - mse: 2.8837e-05 - val_loss: 6.9028e-04 - val_mse: 0.0014
    
    Epoch 00006: val_loss improved from 0.00104 to 0.00069, saving model to /content/checkopint_logs
    Epoch 7/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.2773e-05 - mse: 2.5546e-05 - val_loss: 6.5763e-04 - val_mse: 0.0013
    
    Epoch 00007: val_loss improved from 0.00069 to 0.00066, saving model to /content/checkopint_logs
    Epoch 8/50
    150/150 [==============================] - 2s 16ms/step - loss: 1.0697e-05 - mse: 2.1393e-05 - val_loss: 0.0012 - val_mse: 0.0025
    
    Epoch 00008: val_loss did not improve from 0.00066
    Epoch 9/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.0859e-05 - mse: 2.1718e-05 - val_loss: 0.0020 - val_mse: 0.0040
    
    Epoch 00009: val_loss did not improve from 0.00066
    Epoch 10/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.0822e-05 - mse: 2.1643e-05 - val_loss: 0.0017 - val_mse: 0.0034
    
    Epoch 00010: val_loss did not improve from 0.00066
    Epoch 11/50
    150/150 [==============================] - 2s 15ms/step - loss: 9.4411e-06 - mse: 1.8882e-05 - val_loss: 0.0021 - val_mse: 0.0042
    
    Epoch 00011: val_loss did not improve from 0.00066
    Epoch 12/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.2058e-05 - mse: 2.4116e-05 - val_loss: 0.0012 - val_mse: 0.0024
    
    Epoch 00012: val_loss did not improve from 0.00066
    Epoch 13/50
    150/150 [==============================] - 2s 15ms/step - loss: 8.6561e-06 - mse: 1.7312e-05 - val_loss: 6.4244e-04 - val_mse: 0.0013
    
    Epoch 00013: val_loss improved from 0.00066 to 0.00064, saving model to /content/checkopint_logs
    Epoch 14/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.2479e-05 - mse: 2.4958e-05 - val_loss: 7.1549e-04 - val_mse: 0.0014
    
    Epoch 00014: val_loss did not improve from 0.00064
    Epoch 15/50
    150/150 [==============================] - 2s 15ms/step - loss: 8.2748e-06 - mse: 1.6550e-05 - val_loss: 0.0014 - val_mse: 0.0028
    
    Epoch 00015: val_loss did not improve from 0.00064
    Epoch 16/50
    150/150 [==============================] - 2s 15ms/step - loss: 9.2826e-06 - mse: 1.8565e-05 - val_loss: 0.0016 - val_mse: 0.0031
    
    Epoch 00016: val_loss did not improve from 0.00064
    Epoch 17/50
    150/150 [==============================] - 2s 15ms/step - loss: 8.8093e-06 - mse: 1.7619e-05 - val_loss: 0.0019 - val_mse: 0.0039
    
    Epoch 00017: val_loss did not improve from 0.00064
    Epoch 18/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.1948e-05 - mse: 2.3895e-05 - val_loss: 0.0020 - val_mse: 0.0041
    
    Epoch 00018: val_loss did not improve from 0.00064
    Epoch 19/50
    150/150 [==============================] - 2s 15ms/step - loss: 8.8664e-06 - mse: 1.7733e-05 - val_loss: 0.0011 - val_mse: 0.0023
    
    Epoch 00019: val_loss did not improve from 0.00064
    Epoch 20/50
    150/150 [==============================] - 2s 15ms/step - loss: 7.5307e-06 - mse: 1.5061e-05 - val_loss: 0.0016 - val_mse: 0.0031
    
    Epoch 00020: val_loss did not improve from 0.00064
    Epoch 21/50
    150/150 [==============================] - 2s 15ms/step - loss: 6.8981e-06 - mse: 1.3796e-05 - val_loss: 0.0011 - val_mse: 0.0022
    
    Epoch 00021: val_loss did not improve from 0.00064
    Epoch 22/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.2022e-05 - mse: 2.4044e-05 - val_loss: 0.0013 - val_mse: 0.0026
    
    Epoch 00022: val_loss did not improve from 0.00064
    Epoch 23/50
    150/150 [==============================] - 2s 15ms/step - loss: 1.2670e-05 - mse: 2.5340e-05 - val_loss: 0.0021 - val_mse: 0.0042
    
    Epoch 00023: val_loss did not improve from 0.00064
    


```python
model.load_weights('/content/checkopint_logs')
```




    <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f43bd19ac10>




```python
pred = model.predict(test_data)
```


```python
pred.shape
```




    (1180, 1)


- 주가 예측 시각화

```python
plt.figure(figsize = (12, 9))
plt.plot(np.asarray(y_test)[20:], label = 'actual')
plt.plot(pred, label = 'prediction')
plt.legend()
plt.show()
```


    
![output_25_0](https://user-images.githubusercontent.com/86095931/133551551-8b82cbc4-503b-4fd2-b40a-66e15655e319.png)
    



```python

```
