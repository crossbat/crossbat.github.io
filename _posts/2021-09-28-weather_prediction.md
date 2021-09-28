---
layout : post
title : 'Weather Prediction'
---

시계열 데이터셋 만들기를 하지 못해서 애먹었던 것이다. 이제서야 했다.

- 라이브러리 불러오기

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

```

 - 데이터 불러오기

```python
data_dir = 'C:/Users/Windows10/Desktop/코딩/data/jena_climate_2009_2016.csv'
```


```python
df = pd.read_csv(data_dir)
```

- 데이터 성분 확인

```python
df.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p (mbar)</th>
      <td>420551.0</td>
      <td>989.212776</td>
      <td>8.358481</td>
      <td>913.60</td>
      <td>984.20</td>
      <td>989.58</td>
      <td>994.72</td>
      <td>1015.35</td>
    </tr>
    <tr>
      <th>T (degC)</th>
      <td>420551.0</td>
      <td>9.450147</td>
      <td>8.423365</td>
      <td>-23.01</td>
      <td>3.36</td>
      <td>9.42</td>
      <td>15.47</td>
      <td>37.28</td>
    </tr>
    <tr>
      <th>Tpot (K)</th>
      <td>420551.0</td>
      <td>283.492743</td>
      <td>8.504471</td>
      <td>250.60</td>
      <td>277.43</td>
      <td>283.47</td>
      <td>289.53</td>
      <td>311.34</td>
    </tr>
    <tr>
      <th>Tdew (degC)</th>
      <td>420551.0</td>
      <td>4.955854</td>
      <td>6.730674</td>
      <td>-25.01</td>
      <td>0.24</td>
      <td>5.22</td>
      <td>10.07</td>
      <td>23.11</td>
    </tr>
    <tr>
      <th>rh (%)</th>
      <td>420551.0</td>
      <td>76.008259</td>
      <td>16.476175</td>
      <td>12.95</td>
      <td>65.21</td>
      <td>79.30</td>
      <td>89.40</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>VPmax (mbar)</th>
      <td>420551.0</td>
      <td>13.576251</td>
      <td>7.739020</td>
      <td>0.95</td>
      <td>7.78</td>
      <td>11.82</td>
      <td>17.60</td>
      <td>63.77</td>
    </tr>
    <tr>
      <th>VPact (mbar)</th>
      <td>420551.0</td>
      <td>9.533756</td>
      <td>4.184164</td>
      <td>0.79</td>
      <td>6.21</td>
      <td>8.86</td>
      <td>12.35</td>
      <td>28.32</td>
    </tr>
    <tr>
      <th>VPdef (mbar)</th>
      <td>420551.0</td>
      <td>4.042412</td>
      <td>4.896851</td>
      <td>0.00</td>
      <td>0.87</td>
      <td>2.19</td>
      <td>5.30</td>
      <td>46.01</td>
    </tr>
    <tr>
      <th>sh (g/kg)</th>
      <td>420551.0</td>
      <td>6.022408</td>
      <td>2.656139</td>
      <td>0.50</td>
      <td>3.92</td>
      <td>5.59</td>
      <td>7.80</td>
      <td>18.13</td>
    </tr>
    <tr>
      <th>H2OC (mmol/mol)</th>
      <td>420551.0</td>
      <td>9.640223</td>
      <td>4.235395</td>
      <td>0.80</td>
      <td>6.29</td>
      <td>8.96</td>
      <td>12.49</td>
      <td>28.82</td>
    </tr>
    <tr>
      <th>rho (g/m**3)</th>
      <td>420551.0</td>
      <td>1216.062748</td>
      <td>39.975208</td>
      <td>1059.45</td>
      <td>1187.49</td>
      <td>1213.79</td>
      <td>1242.77</td>
      <td>1393.54</td>
    </tr>
    <tr>
      <th>wv (m/s)</th>
      <td>420551.0</td>
      <td>1.702224</td>
      <td>65.446714</td>
      <td>-9999.00</td>
      <td>0.99</td>
      <td>1.76</td>
      <td>2.86</td>
      <td>28.49</td>
    </tr>
    <tr>
      <th>max. wv (m/s)</th>
      <td>420551.0</td>
      <td>3.056555</td>
      <td>69.016932</td>
      <td>-9999.00</td>
      <td>1.76</td>
      <td>2.96</td>
      <td>4.74</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>wd (deg)</th>
      <td>420551.0</td>
      <td>174.743738</td>
      <td>86.681693</td>
      <td>0.00</td>
      <td>124.90</td>
      <td>198.10</td>
      <td>234.10</td>
      <td>360.00</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(df.columns)
```

    Index(['Date Time', 'p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)',
           'rh (%)', 'VPmax (mbar)', 'VPact (mbar)', 'VPdef (mbar)', 'sh (g/kg)',
           'H2OC (mmol/mol)', 'rho (g/m**3)', 'wv (m/s)', 'max. wv (m/s)',
           'wd (deg)'],
          dtype='object')
    
- 데이터 상관관계 표 나타내기(plt.matshow)

```python
plt.matshow(df.corr())
plt.xticks(range(df.shape[1]), df.columns, rotation = 90)
plt.yticks(range(df.shape[1]), df.columns)

```




    ([<matplotlib.axis.YTick at 0x2577476f7c0>,
      <matplotlib.axis.YTick at 0x2577476f3a0>,
      <matplotlib.axis.YTick at 0x25774aab790>,
      <matplotlib.axis.YTick at 0x25774a080d0>,
      <matplotlib.axis.YTick at 0x25774a085e0>,
      <matplotlib.axis.YTick at 0x25774a08af0>,
      <matplotlib.axis.YTick at 0x25774a0e070>,
      <matplotlib.axis.YTick at 0x25774a08820>,
      <matplotlib.axis.YTick at 0x25774a027f0>,
      <matplotlib.axis.YTick at 0x257749f6bb0>,
      <matplotlib.axis.YTick at 0x25774a0e610>,
      <matplotlib.axis.YTick at 0x25774a0eb20>,
      <matplotlib.axis.YTick at 0x25774a14070>,
      <matplotlib.axis.YTick at 0x25774a14580>,
      <matplotlib.axis.YTick at 0x25774a14a90>],
     [Text(0, 0, 'Date Time'),
      Text(0, 1, 'p (mbar)'),
      Text(0, 2, 'T (degC)'),
      Text(0, 3, 'Tpot (K)'),
      Text(0, 4, 'Tdew (degC)'),
      Text(0, 5, 'rh (%)'),
      Text(0, 6, 'VPmax (mbar)'),
      Text(0, 7, 'VPact (mbar)'),
      Text(0, 8, 'VPdef (mbar)'),
      Text(0, 9, 'sh (g/kg)'),
      Text(0, 10, 'H2OC (mmol/mol)'),
      Text(0, 11, 'rho (g/m**3)'),
      Text(0, 12, 'wv (m/s)'),
      Text(0, 13, 'max. wv (m/s)'),
      Text(0, 14, 'wd (deg)')])




    
![output_5_1](https://user-images.githubusercontent.com/86095931/135032374-113c1c17-ed20-4be0-9d1f-ceeab51ac8e3.png)

    

- 쓰지 않을 데이터는 따로 빼놓기

```python
df_dropped = df.drop(['Tdew (degC)', 'H2OC (mmol/mol)', 'rho (g/m**3)'], axis = 1)
```


```python
df_dropped = df_dropped.set_index(df['Date Time'])
df_dropped = df_dropped.drop('Date Time', axis = 1)
df_dropped
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
      <th>p (mbar)</th>
      <th>T (degC)</th>
      <th>Tpot (K)</th>
      <th>rh (%)</th>
      <th>VPmax (mbar)</th>
      <th>VPact (mbar)</th>
      <th>VPdef (mbar)</th>
      <th>sh (g/kg)</th>
      <th>wv (m/s)</th>
      <th>max. wv (m/s)</th>
      <th>wd (deg)</th>
    </tr>
    <tr>
      <th>Date Time</th>
      <th></th>
      <th></th>
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
      <th>01.01.2009 00:10:00</th>
      <td>996.52</td>
      <td>-8.02</td>
      <td>265.40</td>
      <td>93.30</td>
      <td>3.33</td>
      <td>3.11</td>
      <td>0.22</td>
      <td>1.94</td>
      <td>1.03</td>
      <td>1.75</td>
      <td>152.3</td>
    </tr>
    <tr>
      <th>01.01.2009 00:20:00</th>
      <td>996.57</td>
      <td>-8.41</td>
      <td>265.01</td>
      <td>93.40</td>
      <td>3.23</td>
      <td>3.02</td>
      <td>0.21</td>
      <td>1.89</td>
      <td>0.72</td>
      <td>1.50</td>
      <td>136.1</td>
    </tr>
    <tr>
      <th>01.01.2009 00:30:00</th>
      <td>996.53</td>
      <td>-8.51</td>
      <td>264.91</td>
      <td>93.90</td>
      <td>3.21</td>
      <td>3.01</td>
      <td>0.20</td>
      <td>1.88</td>
      <td>0.19</td>
      <td>0.63</td>
      <td>171.6</td>
    </tr>
    <tr>
      <th>01.01.2009 00:40:00</th>
      <td>996.51</td>
      <td>-8.31</td>
      <td>265.12</td>
      <td>94.20</td>
      <td>3.26</td>
      <td>3.07</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>0.34</td>
      <td>0.50</td>
      <td>198.0</td>
    </tr>
    <tr>
      <th>01.01.2009 00:50:00</th>
      <td>996.51</td>
      <td>-8.27</td>
      <td>265.15</td>
      <td>94.10</td>
      <td>3.27</td>
      <td>3.08</td>
      <td>0.19</td>
      <td>1.92</td>
      <td>0.32</td>
      <td>0.63</td>
      <td>214.3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>31.12.2016 23:20:00</th>
      <td>1000.07</td>
      <td>-4.05</td>
      <td>269.10</td>
      <td>73.10</td>
      <td>4.52</td>
      <td>3.30</td>
      <td>1.22</td>
      <td>2.06</td>
      <td>0.67</td>
      <td>1.52</td>
      <td>240.0</td>
    </tr>
    <tr>
      <th>31.12.2016 23:30:00</th>
      <td>999.93</td>
      <td>-3.35</td>
      <td>269.81</td>
      <td>69.71</td>
      <td>4.77</td>
      <td>3.32</td>
      <td>1.44</td>
      <td>2.07</td>
      <td>1.14</td>
      <td>1.92</td>
      <td>234.3</td>
    </tr>
    <tr>
      <th>31.12.2016 23:40:00</th>
      <td>999.82</td>
      <td>-3.16</td>
      <td>270.01</td>
      <td>67.91</td>
      <td>4.84</td>
      <td>3.28</td>
      <td>1.55</td>
      <td>2.05</td>
      <td>1.08</td>
      <td>2.00</td>
      <td>215.2</td>
    </tr>
    <tr>
      <th>31.12.2016 23:50:00</th>
      <td>999.81</td>
      <td>-4.23</td>
      <td>268.94</td>
      <td>71.80</td>
      <td>4.46</td>
      <td>3.20</td>
      <td>1.26</td>
      <td>1.99</td>
      <td>1.49</td>
      <td>2.16</td>
      <td>225.8</td>
    </tr>
    <tr>
      <th>01.01.2017 00:00:00</th>
      <td>999.82</td>
      <td>-4.82</td>
      <td>268.36</td>
      <td>75.70</td>
      <td>4.27</td>
      <td>3.23</td>
      <td>1.04</td>
      <td>2.01</td>
      <td>1.23</td>
      <td>1.96</td>
      <td>184.9</td>
    </tr>
  </tbody>
</table>
<p>420551 rows × 11 columns</p>
</div>




```python
df_dropped.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p (mbar)</th>
      <td>420551.0</td>
      <td>989.212776</td>
      <td>8.358481</td>
      <td>913.60</td>
      <td>984.20</td>
      <td>989.58</td>
      <td>994.72</td>
      <td>1015.35</td>
    </tr>
    <tr>
      <th>T (degC)</th>
      <td>420551.0</td>
      <td>9.450147</td>
      <td>8.423365</td>
      <td>-23.01</td>
      <td>3.36</td>
      <td>9.42</td>
      <td>15.47</td>
      <td>37.28</td>
    </tr>
    <tr>
      <th>Tpot (K)</th>
      <td>420551.0</td>
      <td>283.492743</td>
      <td>8.504471</td>
      <td>250.60</td>
      <td>277.43</td>
      <td>283.47</td>
      <td>289.53</td>
      <td>311.34</td>
    </tr>
    <tr>
      <th>rh (%)</th>
      <td>420551.0</td>
      <td>76.008259</td>
      <td>16.476175</td>
      <td>12.95</td>
      <td>65.21</td>
      <td>79.30</td>
      <td>89.40</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>VPmax (mbar)</th>
      <td>420551.0</td>
      <td>13.576251</td>
      <td>7.739020</td>
      <td>0.95</td>
      <td>7.78</td>
      <td>11.82</td>
      <td>17.60</td>
      <td>63.77</td>
    </tr>
    <tr>
      <th>VPact (mbar)</th>
      <td>420551.0</td>
      <td>9.533756</td>
      <td>4.184164</td>
      <td>0.79</td>
      <td>6.21</td>
      <td>8.86</td>
      <td>12.35</td>
      <td>28.32</td>
    </tr>
    <tr>
      <th>VPdef (mbar)</th>
      <td>420551.0</td>
      <td>4.042412</td>
      <td>4.896851</td>
      <td>0.00</td>
      <td>0.87</td>
      <td>2.19</td>
      <td>5.30</td>
      <td>46.01</td>
    </tr>
    <tr>
      <th>sh (g/kg)</th>
      <td>420551.0</td>
      <td>6.022408</td>
      <td>2.656139</td>
      <td>0.50</td>
      <td>3.92</td>
      <td>5.59</td>
      <td>7.80</td>
      <td>18.13</td>
    </tr>
    <tr>
      <th>wv (m/s)</th>
      <td>420551.0</td>
      <td>1.702224</td>
      <td>65.446714</td>
      <td>-9999.00</td>
      <td>0.99</td>
      <td>1.76</td>
      <td>2.86</td>
      <td>28.49</td>
    </tr>
    <tr>
      <th>max. wv (m/s)</th>
      <td>420551.0</td>
      <td>3.056555</td>
      <td>69.016932</td>
      <td>-9999.00</td>
      <td>1.76</td>
      <td>2.96</td>
      <td>4.74</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>wd (deg)</th>
      <td>420551.0</td>
      <td>174.743738</td>
      <td>86.681693</td>
      <td>0.00</td>
      <td>124.90</td>
      <td>198.10</td>
      <td>234.10</td>
      <td>360.00</td>
    </tr>
  </tbody>
</table>
</div>

- 바람(풍속, 풍향) -9999.0 

```python
df_dropped['wv (m/s)'] = df_dropped['wv (m/s)'].replace(-9999.0, 0.0)
df_dropped['max. wv (m/s)'] = df_dropped['max. wv (m/s)'].replace(-9999.0, 0.0)

df_dropped.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>p (mbar)</th>
      <td>420551.0</td>
      <td>989.212776</td>
      <td>8.358481</td>
      <td>913.60</td>
      <td>984.20</td>
      <td>989.58</td>
      <td>994.72</td>
      <td>1015.35</td>
    </tr>
    <tr>
      <th>T (degC)</th>
      <td>420551.0</td>
      <td>9.450147</td>
      <td>8.423365</td>
      <td>-23.01</td>
      <td>3.36</td>
      <td>9.42</td>
      <td>15.47</td>
      <td>37.28</td>
    </tr>
    <tr>
      <th>Tpot (K)</th>
      <td>420551.0</td>
      <td>283.492743</td>
      <td>8.504471</td>
      <td>250.60</td>
      <td>277.43</td>
      <td>283.47</td>
      <td>289.53</td>
      <td>311.34</td>
    </tr>
    <tr>
      <th>rh (%)</th>
      <td>420551.0</td>
      <td>76.008259</td>
      <td>16.476175</td>
      <td>12.95</td>
      <td>65.21</td>
      <td>79.30</td>
      <td>89.40</td>
      <td>100.00</td>
    </tr>
    <tr>
      <th>VPmax (mbar)</th>
      <td>420551.0</td>
      <td>13.576251</td>
      <td>7.739020</td>
      <td>0.95</td>
      <td>7.78</td>
      <td>11.82</td>
      <td>17.60</td>
      <td>63.77</td>
    </tr>
    <tr>
      <th>VPact (mbar)</th>
      <td>420551.0</td>
      <td>9.533756</td>
      <td>4.184164</td>
      <td>0.79</td>
      <td>6.21</td>
      <td>8.86</td>
      <td>12.35</td>
      <td>28.32</td>
    </tr>
    <tr>
      <th>VPdef (mbar)</th>
      <td>420551.0</td>
      <td>4.042412</td>
      <td>4.896851</td>
      <td>0.00</td>
      <td>0.87</td>
      <td>2.19</td>
      <td>5.30</td>
      <td>46.01</td>
    </tr>
    <tr>
      <th>sh (g/kg)</th>
      <td>420551.0</td>
      <td>6.022408</td>
      <td>2.656139</td>
      <td>0.50</td>
      <td>3.92</td>
      <td>5.59</td>
      <td>7.80</td>
      <td>18.13</td>
    </tr>
    <tr>
      <th>wv (m/s)</th>
      <td>420551.0</td>
      <td>2.130191</td>
      <td>1.542334</td>
      <td>0.00</td>
      <td>0.99</td>
      <td>1.76</td>
      <td>2.86</td>
      <td>28.49</td>
    </tr>
    <tr>
      <th>max. wv (m/s)</th>
      <td>420551.0</td>
      <td>3.532074</td>
      <td>2.340482</td>
      <td>0.00</td>
      <td>1.76</td>
      <td>2.96</td>
      <td>4.74</td>
      <td>23.50</td>
    </tr>
    <tr>
      <th>wd (deg)</th>
      <td>420551.0</td>
      <td>174.743738</td>
      <td>86.681693</td>
      <td>0.00</td>
      <td>124.90</td>
      <td>198.10</td>
      <td>234.10</td>
      <td>360.00</td>
    </tr>
  </tbody>
</table>
</div>


- 데이터 분리

```python
split = int(df_dropped.shape[0]*0.7)

train_data = df_dropped.iloc[:split]
test_data = df_dropped.iloc[split:]
```

-데이터 정제

```python
class preprocessing:
    def __init__(self):
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std

    def normalization(data):
        mean = data.mean(axis = 0)
        std = data.std(axis = 0)
        return (data - mean) / std

    def MinMaxscaler(data):
        min = data.min(axis = 0)
        max = data.max(axis = 0)
        return (data - min) / (max - min)
```


```python
train_scaled = preprocessing.MinMaxscaler(train_data)
```


```python
train_scaled
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
      <th>p (mbar)</th>
      <th>T (degC)</th>
      <th>Tpot (K)</th>
      <th>rh (%)</th>
      <th>VPmax (mbar)</th>
      <th>VPact (mbar)</th>
      <th>VPdef (mbar)</th>
      <th>sh (g/kg)</th>
      <th>wv (m/s)</th>
      <th>max. wv (m/s)</th>
      <th>wd (deg)</th>
    </tr>
    <tr>
      <th>Date Time</th>
      <th></th>
      <th></th>
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
      <th>01.01.2009 00:10:00</th>
      <td>0.835550</td>
      <td>0.254629</td>
      <td>0.248031</td>
      <td>0.923033</td>
      <td>0.040985</td>
      <td>0.084272</td>
      <td>0.005223</td>
      <td>0.081679</td>
      <td>0.070403</td>
      <td>0.074468</td>
      <td>0.423056</td>
    </tr>
    <tr>
      <th>01.01.2009 00:20:00</th>
      <td>0.836054</td>
      <td>0.248004</td>
      <td>0.241495</td>
      <td>0.924182</td>
      <td>0.039263</td>
      <td>0.081003</td>
      <td>0.004986</td>
      <td>0.078843</td>
      <td>0.049214</td>
      <td>0.063830</td>
      <td>0.378056</td>
    </tr>
    <tr>
      <th>01.01.2009 00:30:00</th>
      <td>0.835651</td>
      <td>0.246305</td>
      <td>0.239819</td>
      <td>0.929925</td>
      <td>0.038919</td>
      <td>0.080639</td>
      <td>0.004748</td>
      <td>0.078276</td>
      <td>0.012987</td>
      <td>0.026809</td>
      <td>0.476667</td>
    </tr>
    <tr>
      <th>01.01.2009 00:40:00</th>
      <td>0.835449</td>
      <td>0.249703</td>
      <td>0.243338</td>
      <td>0.933372</td>
      <td>0.039780</td>
      <td>0.082819</td>
      <td>0.004511</td>
      <td>0.080545</td>
      <td>0.023240</td>
      <td>0.021277</td>
      <td>0.550000</td>
    </tr>
    <tr>
      <th>01.01.2009 00:50:00</th>
      <td>0.835449</td>
      <td>0.250382</td>
      <td>0.243841</td>
      <td>0.932223</td>
      <td>0.039952</td>
      <td>0.083182</td>
      <td>0.004511</td>
      <td>0.080545</td>
      <td>0.021873</td>
      <td>0.026809</td>
      <td>0.595278</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>05.08.2014 01:40:00</th>
      <td>0.771665</td>
      <td>0.638356</td>
      <td>0.635830</td>
      <td>0.940264</td>
      <td>0.269675</td>
      <td>0.543407</td>
      <td>0.020418</td>
      <td>0.536018</td>
      <td>0.123718</td>
      <td>0.108936</td>
      <td>0.644722</td>
    </tr>
    <tr>
      <th>05.08.2014 01:50:00</th>
      <td>0.773982</td>
      <td>0.637676</td>
      <td>0.634825</td>
      <td>0.949454</td>
      <td>0.268986</td>
      <td>0.546676</td>
      <td>0.017331</td>
      <td>0.539421</td>
      <td>0.121668</td>
      <td>0.122553</td>
      <td>0.640278</td>
    </tr>
    <tr>
      <th>05.08.2014 02:00:00</th>
      <td>0.775091</td>
      <td>0.635128</td>
      <td>0.632143</td>
      <td>0.951752</td>
      <td>0.266230</td>
      <td>0.542317</td>
      <td>0.016382</td>
      <td>0.534884</td>
      <td>0.175666</td>
      <td>0.141277</td>
      <td>0.693333</td>
    </tr>
    <tr>
      <th>05.08.2014 02:10:00</th>
      <td>0.774184</td>
      <td>0.636487</td>
      <td>0.633652</td>
      <td>0.952901</td>
      <td>0.267608</td>
      <td>0.545950</td>
      <td>0.016144</td>
      <td>0.538287</td>
      <td>0.099795</td>
      <td>0.094468</td>
      <td>0.700278</td>
    </tr>
    <tr>
      <th>05.08.2014 02:20:00</th>
      <td>0.773378</td>
      <td>0.634109</td>
      <td>0.631473</td>
      <td>0.949454</td>
      <td>0.265025</td>
      <td>0.539048</td>
      <td>0.017094</td>
      <td>0.531480</td>
      <td>0.046480</td>
      <td>0.057872</td>
      <td>0.602222</td>
    </tr>
  </tbody>
</table>
<p>294385 rows × 11 columns</p>
</div>


- 데이터 분할(훈련, 검증)

```python
train_data = train_scaled.iloc[:int(train_scaled.shape[0] * 0.7)]
val_data = train_scaled.iloc[int(train_scaled.shape[0] * 0.7):]
```

- x, y 데이터 분리

```python
def x_y_divid(data, target_cols):
    x_data = data.drop(target_cols, axis = 1)
    y_data = data[target_cols]

    return x_data, y_data
```


```python
x_train, y_train = x_y_divid(data = train_data, target_cols= 'T (degC)')
print(x_train.head())
print('='*20)
print(y_train.head())
```

                         p (mbar)  Tpot (K)    rh (%)  VPmax (mbar)  VPact (mbar)  \
    Date Time                                                                       
    01.01.2009 00:10:00  0.835550  0.248031  0.923033      0.040985      0.084272   
    01.01.2009 00:20:00  0.836054  0.241495  0.924182      0.039263      0.081003   
    01.01.2009 00:30:00  0.835651  0.239819  0.929925      0.038919      0.080639   
    01.01.2009 00:40:00  0.835449  0.243338  0.933372      0.039780      0.082819   
    01.01.2009 00:50:00  0.835449  0.243841  0.932223      0.039952      0.083182   
    
                         VPdef (mbar)  sh (g/kg)  wv (m/s)  max. wv (m/s)  \
    Date Time                                                               
    01.01.2009 00:10:00      0.005223   0.081679  0.070403       0.074468   
    01.01.2009 00:20:00      0.004986   0.078843  0.049214       0.063830   
    01.01.2009 00:30:00      0.004748   0.078276  0.012987       0.026809   
    01.01.2009 00:40:00      0.004511   0.080545  0.023240       0.021277   
    01.01.2009 00:50:00      0.004511   0.080545  0.021873       0.026809   
    
                         wd (deg)  
    Date Time                      
    01.01.2009 00:10:00  0.423056  
    01.01.2009 00:20:00  0.378056  
    01.01.2009 00:30:00  0.476667  
    01.01.2009 00:40:00  0.550000  
    01.01.2009 00:50:00  0.595278  
    ====================
    Date Time
    01.01.2009 00:10:00    0.254629
    01.01.2009 00:20:00    0.248004
    01.01.2009 00:30:00    0.246305
    01.01.2009 00:40:00    0.249703
    01.01.2009 00:50:00    0.250382
    Name: T (degC), dtype: float64
    


```python
test_scaled = preprocessing.MinMaxscaler(test_data)
```


```python
x_val, y_val = x_y_divid(data = val_data, target_cols= 'T (degC)')
x_test, y_test = x_y_divid(data = test_scaled, target_cols= 'T (degC)')
```

- 데이터셋 함수 만들기

```python
def windowed_dataset(x_data, y_data, window_size, batch_size, shuffle):
    ds_x = tf.data.Dataset.from_tensor_slices(x_data)
    ds_x = ds_x.window(window_size, stride = 1, shift = 1, drop_remainder= True)
    ds_x = ds_x.flat_map(lambda x : x.batch(window_size))

    ds_y = tf.data.Dataset.from_tensor_slices(y_data[window_size:])

    dataset = tf.data.Dataset.zip((ds_x, ds_y))
    return dataset.batch(batch_size).prefetch(1)
```


```python
train_set = windowed_dataset(x_train, y_train, window_size = 288, batch_size = 64, shuffle = True)
val_set = windowed_dataset(x_val, y_val, window_size = 288, batch_size = 64, shuffle = False)
test_set = windowed_dataset(x_test, y_test, window_size = 288, batch_size = 64, shuffle = False)
```


```python
for data in train_set.take(1):
    print(data[0].shape)
```

    (64, 288, 10)
    
- 모델 만들기

```python
model = Sequential([
    Conv1D(128, 5, padding = 'causal', activation = 'relu', input_shape = [288, 10]),
    LSTM(64, activation = 'tanh'),
    Dense(32, activation = 'relu'),
    Dense(16, activation = 'relu'),
    Dense(1)
])
```


```python
model.compile(loss= tf.keras.losses.Huber(), optimizer = 'adam', metrics = ['mse'])
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_1 (Conv1D)            (None, 288, 128)          6528      
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 64)                49408     
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                2080      
    _________________________________________________________________
    dense_4 (Dense)              (None, 16)                528       
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 58,561
    Trainable params: 58,561
    Non-trainable params: 0
    _________________________________________________________________
    
- 콜벡 함수 불러오기

```python
mc = ModelCheckpoint(filepath = 'C:/Users/Windows10/Desktop/pythonProject/Git/logs/weather', monitor = 'val_loss', save_best_only= True, save_weights_only= True, save_freq= 'epoch')
es = EarlyStopping(monitor= 'val_loss', patience= 5)
```

- 모델 실행

```python
history = model.fit(train_set, epochs= 50, batch_size = 64, callbacks= [mc, es], validation_data = val_set)
```

    Epoch 1/50
    3216/3216 [==============================] - 832s 258ms/step - loss: 4.0135e-04 - mse: 8.0269e-04 - val_loss: 3.0836e-04 - val_mse: 6.1673e-04
    Epoch 2/50
    3216/3216 [==============================] - 828s 257ms/step - loss: 9.0935e-05 - mse: 1.8187e-04 - val_loss: 9.3921e-04 - val_mse: 0.0019
    Epoch 3/50
    3216/3216 [==============================] - 831s 258ms/step - loss: 6.3121e-05 - mse: 1.2624e-04 - val_loss: 2.7443e-04 - val_mse: 5.4887e-04
    Epoch 4/50
    3216/3216 [==============================] - 827s 257ms/step - loss: 5.0628e-05 - mse: 1.0126e-04 - val_loss: 3.6077e-04 - val_mse: 7.2154e-04
    Epoch 5/50
    3216/3216 [==============================] - 835s 260ms/step - loss: 4.1755e-05 - mse: 8.3510e-05 - val_loss: 2.6353e-04 - val_mse: 5.2705e-04
    Epoch 6/50
    3216/3216 [==============================] - 830s 258ms/step - loss: 3.9000e-05 - mse: 7.7999e-05 - val_loss: 4.6934e-04 - val_mse: 9.3868e-04
    Epoch 7/50
    3216/3216 [==============================] - 832s 259ms/step - loss: 3.8830e-05 - mse: 7.7660e-05 - val_loss: 2.0859e-04 - val_mse: 4.1717e-04
    Epoch 8/50
    3216/3216 [==============================] - 1069s 333ms/step - loss: 3.4448e-05 - mse: 6.8896e-05 - val_loss: 1.5965e-04 - val_mse: 3.1930e-04
    Epoch 9/50
    3216/3216 [==============================] - 1050s 326ms/step - loss: 3.1476e-05 - mse: 6.2952e-05 - val_loss: 9.9053e-05 - val_mse: 1.9811e-04
    Epoch 10/50
    3216/3216 [==============================] - 1071s 333ms/step - loss: 2.1904e-05 - mse: 4.3808e-05 - val_loss: 1.4497e-04 - val_mse: 2.8994e-04
    Epoch 11/50
    3216/3216 [==============================] - 849s 264ms/step - loss: 2.3855e-05 - mse: 4.7711e-05 - val_loss: 1.0187e-04 - val_mse: 2.0374e-04
    Epoch 12/50
    3216/3216 [==============================] - 809s 252ms/step - loss: 2.2041e-05 - mse: 4.4082e-05 - val_loss: 1.7669e-04 - val_mse: 3.5338e-04
    Epoch 13/50
    3216/3216 [==============================] - 812s 252ms/step - loss: 2.0518e-05 - mse: 4.1036e-05 - val_loss: 1.2031e-04 - val_mse: 2.4062e-04
    Epoch 14/50
    3216/3216 [==============================] - 807s 251ms/step - loss: 1.8312e-05 - mse: 3.6625e-05 - val_loss: 4.3044e-04 - val_mse: 8.6087e-04
    
    
- 학습 결과(loss, mse)

```python
hist = history.history
epochs = range(len(hist['loss']))

plt.figure(figsize = (16,9))
plt.plot(epochs, hist['loss'], label = 'training loss')
plt.plot(epochs, hist['val_loss'], label = 'validation loss')
plt.title('training and validation loss')
plt.legend()

plt.figure(figsize = (16, 9))
plt.plot(epochs, hist['mse'], label = 'training mse')
plt.plot(epochs, hist['val_mse'], label = 'validation mse')
plt.title('training and validation mse')
plt.legend()

plt.show()

```


    
![output_26_0](https://user-images.githubusercontent.com/86095931/135035129-452a10ae-7ef5-47de-8b44-e0425049b4ef.png)

    



    
![output_26_1](https://user-images.githubusercontent.com/86095931/135035148-15c0e96b-7bb0-48a5-b614-6ed0b23577da.png)

    

- 예측 결과

```python
y_pred = model.predict(test_set)
```


```python
print(y_pred)
print(y_test)
```

    [[0.6006168 ]
     [0.60086715]
     [0.5938302 ]
     ...
     [0.27581054]
     [0.2778709 ]
     [0.26019403]]
    Date Time
    05.08.2014 02:30:00    0.547354
    05.08.2014 02:40:00    0.544815
    05.08.2014 02:50:00    0.540519
    05.08.2014 03:00:00    0.539543
    05.08.2014 03:10:00    0.542082
                             ...   
    31.12.2016 23:20:00    0.192931
    31.12.2016 23:30:00    0.206600
    31.12.2016 23:40:00    0.210310
    31.12.2016 23:50:00    0.189416
    01.01.2017 00:00:00    0.177895
    Name: T (degC), Length: 126166, dtype: float64
    


```python

plt.figure(figsize = (16, 9))
plt.plot(range(len(y_pred)), y_pred, label = 'prediction')
plt.plot(range(len(y_test)), y_test, label = 'actual')
plt.legend()
plt.show()
```


    
![output_29_0](https://user-images.githubusercontent.com/86095931/135035220-be179450-33a5-44a3-ab03-0e67ea90cf86.png)

    

