---
layer : post
title : 'Text Generator'
---


# Text Generator (Tensorflow Tutorial)
- 텐서플로우 튜토리얼을 보고 만들었고, 데이터만 다른 데이터를 사용하였다.

- - -

- 필요 라이브러리 생성


```python
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
```

데이터 다운로드 (프로젝트 구텐베르크 전자책)


```python
data1 = get_file("a_childhood_of_the_orient", 'https://www.gutenberg.org/files/66019/66019-0.txt')
data2 = get_file('MISS LOCHINVAR', 'https://www.gutenberg.org/files/66018/66018-0.txt')
```

    Downloading data from https://www.gutenberg.org/files/66018/66018-0.txt
    311296/304380 [==============================] - 0s 1us/step
    


```python
text1 = open(data1, 'rb').read().decode(encoding = 'utf-8')
text2 = open(data2, 'rb').read().decode(encoding = 'utf-8')
```


```python
#데이터 앞 차례부분 삭제
text1 = text1[3000:]
text2 = text2[3000:]
```


```python
text = text1 + text2
```


```python
len(text)
```




    676936




```python
print(text[:250])
```

    is_, you are five years old. I wish you many
    happy returns of the day.”
    
    He drew up a chair, and sat down by my bed. Carefully unfolding a piece
    of paper, he brought forth a small Greek flag.
    
    “Do you know what this is?”
    
    I nodded.
    
    “Do you
    


```python
vocab = sorted(set(text)) # 단어사전 만들기

char2idx = {u : i for i, u in enumerate(vocab)} # 문자 -> 숫자
idx2char = np.array(vocab)                      # 숫자 -> 문자
```


```python
for char, _ in zip(char2idx, range(20)):
  print(char, char2idx[char])
```

    
     0
     1
      2
    ! 3
    " 4
    $ 5
    % 6
    & 7
    ' 8
    ( 9
    ) 10
    * 11
    , 12
    - 13
    . 14
    / 15
    0 16
    1 17
    2 18
    3 19
    


```python
text_as_int = [char2idx[c] for c in text]
text_as_int[:5]
```




    [66, 76, 57, 12, 2]




```python
seq_len = 100 # 단어 뭉치
examples_per_epoch = len(text) // seq_len # ?

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(10):
  print(idx2char[i])
```

    i
    s
    _
    ,
     
    y
    o
    u
     
    a
    


```python
# 단어 뭉치를 기준으로 데이터 나누기
seq = char_dataset.batch(seq_len +1 , drop_remainder= True)

for item in seq.take(5):
  print(repr(''.join(idx2char[item.numpy()])))
```

    'is_, you are five years old. I wish you many\r\nhappy returns of the day.”\r\n\r\nHe drew up a chair, and s'
    'at down by my bed. Carefully unfolding a piece\r\nof paper, he brought forth a small Greek flag.\r\n\r\n“Do'
    ' you know what this is?”\r\n\r\nI nodded.\r\n\r\n“Do you know what it stands for?”\r\n\r\nBefore I could think of'
    ' an adequate reply, he leaned toward me and said\r\nearnestly, his fiery black eyes holding mine:\r\n\r\n“I'
    't stands for the highest civilization the world has ever known. It\r\nstands for Greece, who has taught'
    


```python
# 입력 데이터와 출력 데이터 나누기
def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = seq.map(split_input_target)
```


```python
batch_size = 64
buffer_size = 10000

# Buffer_size만큼 섞고, batch_size크기로 데이터 나누기
dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder = True)
```


```python
vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 1024
```


```python
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = Sequential([
                      Embedding(vocab_size, embedding_dim, batch_input_shape = [batch_size, None]),
                      LSTM(rnn_units, return_sequences= True, stateful= True, recurrent_initializer= 'glorot_uniform'),
                      Dense(vocab_size)
  ])
  return model
```


```python
model = build_model(vocab_size = vocab_size, embedding_dim= embedding_dim, rnn_units= rnn_units, batch_size = batch_size)
```


```python
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_prediction = model(input_example_batch)
  print(example_batch_prediction.shape)
```

    (64, 100, 98)
    


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (64, None, 256)           25088     
    _________________________________________________________________
    lstm (LSTM)                  (64, None, 1024)          5246976   
    _________________________________________________________________
    dense (Dense)                (64, None, 98)            100450    
    =================================================================
    Total params: 5,372,514
    Trainable params: 5,372,514
    Non-trainable params: 0
    _________________________________________________________________
    


```python
sampled_indices = tf.random.categorical(example_batch_prediction[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis = -1).numpy()
```


```python
print(sampled_indices)
```

    [ 4 43 23 11 24 18 69  3 55 12 35 86 43 12 18 37 84 22 16 43 52 95 74 60
     12 46 88 38 68 82 57  8 35 12 33  8 56 39 37 67 51 74 40 33 45 71 78 77
     68 48 65 52 81 27 33  0 91  1 53 13 66 24 69 34  6 71 21 12 93 83 85 11
     61 95 30 89 38 16 43 86  4  5 54 76 70 69 38  9 93 65 22 64  3 22 12 47
      9 28 43 23]
    


```python
print('input:',repr(''.join(idx2char[input_example_batch[0]])))
print('expected :', repr(''.join(idx2char[sampled_indices])))
```

    input: 'e\r\n_parti_ if he had money and position, irrespective of any other\r\nqualifications.\r\n\r\nFor a long ti'
    expected : '"O7*82l![,GÏO,2IÉ60OX’qc,RæJky_\'G,E\']KIjWqLEQnutkThXx;E\në\rY-i8lF%n5,ôzË*d’BèJ0OÏ"$ZsmlJ(ôh6g!6,S(?O7'
    


```python
# 손실함수 지정
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits = True)

example_batch_loss = loss(target_example_batch, example_batch_prediction)

```


```python
model.compile(optimizer = 'adam', loss = loss)
```


```python
# callback 함수 지정
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_ {epoch}')

checkpoint_callback = ModelCheckpoint(filepath= checkpoint_prefix, save_weights_only= True)
```


```python
epochs = 20

history = model.fit(dataset, epochs = epochs, callbacks=[checkpoint_callback])
```

    Epoch 1/20
    104/104 [==============================] - 10s 69ms/step - loss: 2.8178
    Epoch 2/20
    104/104 [==============================] - 8s 70ms/step - loss: 2.1709
    Epoch 3/20
    104/104 [==============================] - 8s 72ms/step - loss: 1.8932
    Epoch 4/20
    104/104 [==============================] - 8s 73ms/step - loss: 1.7058
    Epoch 5/20
    104/104 [==============================] - 8s 73ms/step - loss: 1.5750
    Epoch 6/20
    104/104 [==============================] - 8s 72ms/step - loss: 1.4832
    Epoch 7/20
    104/104 [==============================] - 8s 71ms/step - loss: 1.4122
    Epoch 8/20
    104/104 [==============================] - 8s 71ms/step - loss: 1.3576
    Epoch 9/20
    104/104 [==============================] - 8s 70ms/step - loss: 1.3101
    Epoch 10/20
    104/104 [==============================] - 8s 70ms/step - loss: 1.2681
    Epoch 11/20
    104/104 [==============================] - 8s 71ms/step - loss: 1.2301
    Epoch 12/20
    104/104 [==============================] - 8s 71ms/step - loss: 1.1933
    Epoch 13/20
    104/104 [==============================] - 8s 72ms/step - loss: 1.1573
    Epoch 14/20
    104/104 [==============================] - 8s 72ms/step - loss: 1.1222
    Epoch 15/20
    104/104 [==============================] - 8s 72ms/step - loss: 1.0856
    Epoch 16/20
    104/104 [==============================] - 8s 71ms/step - loss: 1.0478
    Epoch 17/20
    104/104 [==============================] - 8s 70ms/step - loss: 1.0104
    Epoch 18/20
    104/104 [==============================] - 8s 70ms/step - loss: 0.9698
    Epoch 19/20
    104/104 [==============================] - 8s 71ms/step - loss: 0.9314
    Epoch 20/20
    104/104 [==============================] - 8s 71ms/step - loss: 0.8900
    

- 데이터 출력


```python
#예측 단계를 간단하게 유지하기 위해 배치 사이즈를 1로 함
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size = 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (1, None, 256)            25088     
    _________________________________________________________________
    lstm_1 (LSTM)                (1, None, 1024)           5246976   
    _________________________________________________________________
    dense_1 (Dense)              (1, None, 98)             100450    
    =================================================================
    Total params: 5,372,514
    Trainable params: 5,372,514
    Non-trainable params: 0
    _________________________________________________________________
    


```python
def generate_text(model, start_string):
  num_generate = 10000 # 생성할 문자의 수

  input_eval = [char2idx[s] for s in start_string] # 문자 벡터화
  input_eval = tf.expand_dims(input_eval, 0)

  text_generated = [] # 결과 저장용 리스트

  temperature = 1.0
# 온도가 낮으면 예측 가능한 텍스트가,
# 높으면 의외의 텍스트가 됨

# 상태 초기화
  model.reset_states() #배치 사이즈 = 1
  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0)

    predictions = predictions / temperature
    predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1, 0].numpy()

    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))
```


```python
print(generate_text(model, start_string= u'The youngest member'))
```

    The youngest members of Poop. Bago
    I would not be mostly and fashion, and the day
    
    It is the blue forestors they were cervanted years age my like children of
    Arif But the Hummel the other ways was laigh
    a mindleady predition. “Nou indite her going touch
    your true.”
    
    “But within be present,” I commored him. “Fork meware her
    fact that the school and behered.
    At the epists the man of the first time free
    days. In my heart was not for the children had been moreatest friends, and then
    she was struckly closed, and then buy had phints treath
    of membaring shone mitherly applicable teever myself for
    her trousper. She esurply, the untrain, the heroine
    experience she could not each _chedians. I struck quite failing a
    numsur’ at Themenuen understand the brown sided was covered with
    afternative, and--and I do not see much all offend me.”
    
    “Hang loving!”
    
    “I wonder like about to-say opened.”
    
    “When what to a ntile, if you will be ablulged. Of the
    inCense
    that I had been told that that he asked time up the valiars feor
    attendants breakfasting as they are assidants as life in relief?
    
    [that was the second time a linand abouteful man, I could
    have rese.
    
    “After I heppiness you are right. You at
    once my flished befire, holding many more first person to paps at ouch.
    
    Again I threw hers on its close law I created
    vaintle with this I would call them.”
    
    From Summon Kallering it bent, and to let him te lighted her cousin. Jan
    gave for a ten, that my cousin began to have her every day more than any one to kills and kissed me. It was neither
    then. When the greatest parts is discovered she was wonderful the indifference to the day of the uppar which enthusiasts
    for a fit of thinks before her slaves required to each other he
    to permit tigsting in that man, “why did you see ready feel close or attinent, and its is the
    girl wom which Jan had consulted her heart, and
    when the rooms are to buy part of getting morification
    number or currental way the iprogrchap at Ali Baba. It active. Betray,
    like Daisy was banister without even longer,” said
    Jan, who plan for centerness, far away from since, Jan threw outher
    games to bul the landing, he was a garling for accidance in life-smile. She had not
    a city assame at her. The storm has left beyond him. Besides,
    they were asked whether I could have made her he was her into the villitary post in her efforts there. It was from a sudden’s dog By a
    Pasha, and very little Djimlah only he began some committy becimed
    from Gwen’s head. In a square arment dather to bithers as a
    sigh of working. My book telling me even into life perfect
    guest.
    
    A fortunate animaldaraties.
    
    [Illustration: The monk had always had heard after
    this was frughted about them ever since the Greek revisence, who, lady had
    just harded faver a changes. In the other dawn the
    boards with her at the house alone--it nice at the and on that could be her hair.
    
    “No, I know.”
    
    “If you lent you this--me did ever those Turkish pobrors
    of the world.”
    
    “Gone beyove,” and as Jan ppantly, studied and peaches with reparation for
    superstitions to eat might. But though the books now
    stroke, and this was the class was the soon, but nonsineeally, though there threw off them glad to
    celebrate you!”
    
    “Until y’s heeld, mamma
    use cause. And French lained was perhaps
    at its we denerally absolutely, nor did he turned to the
    Jan’s hour before I had before I could bore or history, ite
    told at reason, the faces and tame to the paper for
    cermining better than both papare we hidden leaving being laderstases, in paragraph 1.E below. “Fract
    Hemethey’ halled the Turks. They had their live,” said Jan, since
    in a man’s-line. The little time in its lone forms aworemones. My intimat
    patriotism at that moment we were at a simple with have y
    come.
    
    “Come! ‘THE 
    THE ASUSS."
    
    
    After A long aside without the books of the effendi, reminding
    them, and in and out of those who could be nice.”
    
    Janet ready to each other nicer than perfectly insilement and her forehead even uncillion as Gladys, having badly we made this purposes pure. Ih all her
    going to reach our nise corner.
    
    Apper to Jan should think, and I tried to be in her even frung farther for me. I was ready to
    have the dim Turks, because she was a little dumble feet, and then Can lets themselves nursery that there was
    so! I meant used to leave Turkey. I could pass meant me. He wished my pass
    was brough with an left the given engult to remember that one around herself to meen my own different forms of which Djimlah and Cena
    Susantly she apped my animal--past her bandage into herself, and were struck
    Jan, who, and we had plensed for an ensity that I should come back again. This it
    was unconscious of attitude toward her.
    
    At that night I only nodded to them.
    
    “Why did you do?” she’d past armin as Kanti, who
    would perchaise them lapped us and lockeds--after all, a rich humalous North over
    More like a large beside.
    After this eBook (amountane long, for
    very about St. Norking or proprie assurance failed with intimate
    and makemanies. “I thinks about beying to rum
    I do not noble you averimentain thought of all seven orders. Sydney
    nelted stop in an unself. I do!” she asked, and at
    Pashatwatedner and Sitanthy
    salutedly. And that he time to your mother the
    production to have Gwen found herself to me.
    
    “What is what you permause she was a girl over the atmosphere. In a few moments he
    would begin amous Viva was the secced to
    her.”
    
    _Wangeriag_. Besides,” he conquered her heart ton, about them.
    It was at the nd found in in her returned form anynotseip mither.
    For is the only snot one sittic, and were found in that time I had for
    swell beautiful and an inside, are so much man, and as I asked
    her cousin Janet. I wished to miss the foreible hour of a little-ticken knew of it, even
    if that we stupend of wore here, and told us at the back riddle coling, and the
    paper. This escended iname up to
    since the
    first thing. I might have had managed to send to him Jan
    to play with Gwen, his head a remorch fram _pathab_.
    
    Home, homeloted considerable as he, they always warm
    affectation again denished merbed in the end was under from the darkness.
    The east wite Fred, I might have been holding that it may
    lead must me entirely as at this furship endicance
    in a brace of minutes niece, you do anything more than speaker, if I kept
    tintestled and make her headerly, Syd, that something for us things and
    have me the wind I’m learn to reprive on that darkness.
    
    eg home, a dolla’s man as a vishit--entill have you here ways in much aller in than little
    finger-shoe for our pehiods. It made a very nightfal
    nest to hear ose profits to help long, while I had to accomplish nationam a bad in exlier. The
    arenity nor prite in checks in the spiders. It
    ensolate Indian for a persona who will ve pown of able to
    edrhanger, and many of them at all eusy
    ton manners through the atmosphere, of the piece of the
    subsline characce is in the station, during this work (all high
    for a minute from sending him half-by, inquiries for us to lose my feither. There we are Allah, the
    nobsessed in out of the room.
    
    From hands--iss.
    
    “Gwendoline, Jund, of bechieve, I thinks, because they had a
    sich girl, not becring mose confident the Graham faith in pungs that she answered nebroly
    recuried by reliefatively. Only Sydney and Mr. Graham forentener? I
    tasted.
    
    “To play you are?”
    
    “No, the music and sainture--as he was more there to Greeks. The baby of
    Gwen’s heart
    at the chocks. I rood only began to repart to never
    the little flag from be bods which age children if they could be able to say it. It always said nothing that
    Miss Lochinvar! To prace Turkish things are the manner. When
    she’ll have less possible.”
    
    Sydney heard. I gazed on the floor,
    and ashe
    parting acting, despitite the a dombination this
    leg.”
    
    She shoold horring them, one of her language in the dugies, to give you
    accident from her did so one afternating any beyind of the
    werthopges. Out of her should should hand
    from her and I could not any longer; then, to show how
    they had never consented--maddanderstand actual. On the saint of provided it was the production whom
    she had discussed to Arif and pressic undished.
    
    “Hall repress little fee that, better.”
    
    “G am hold Dride, with several respinsing emect Gutenberg-tm electronic works
    
    For anything so very so hard was sobt on the sleeping-class on the same time, Schublan, and he has
    preceded the events I
    bevery way unkeptafal past protested associated in accident holdesty-hall be stretce the door
    frest.
    
    It is a pertape,” said Gwen. “Good-time and not tell the extending of this grown-up stock I felt there, and
    my end is unpredumed, incrudiously rose, tracefully stronger, and her mother longed for and on, “the
    young half in such parangate him. or obtained my intream all for a week--you
    wouldn’t say your arrive Foundation is any one. I hear by a rush, where
    I sat down cooly I’d hair to come into the full Project Gutenberg-tm License must call half people are babies to did. I know it became more demiplose in a lot, from
    forth included. Thould take I wouldn’t try for equipment including Project Gutenberg. But at the monastery
    which the young fields were very little for luttle:
    “His poor girls fown about dew song
    thoughtfully. I was still deced them and baded the Greeks
    and year away, and I do ever seen anything but consuried in
    their leaun the pipsible demiliance with this race me.
    I after a long tine then, try that men anguisplessed the compose
    of streegs on her arm. She approached me better than point over, and her eyes
    his arms after they we’d have you back, and print call day from a land, next the glormed
    to match her and kissed his best.
    
    There was a nuisance at Jan quige fervently, since
    it may be and simple as bed at last a leas use. It was only 
    


```python

```
