#!/usr/bin/python
# encoding: utf-8

"""
@author: Ian
@contact:yongguiluo@hotmail.com
@file: attention_test.py
@time: 2019/3/15 9:48
"""
from keras.preprocessing import sequence
from keras.datasets import imdb
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from apps.attention.attention_keras import Attention, Position_Embedding


if __name__ == "__main__":
    max_features = 20000
    maxlen = 80
    batch_size = 32

    print('Loading data...')
    #num_words: max number of words to include
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(max_features, 128)(S_inputs)
    # embeddings = Position_Embedding()(embeddings) # 增加Position_Embedding能轻微提高准确率
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.5)(O_seq)
    outputs = Dense(1, activation='sigmoid')(O_seq)

    model = Model(inputs=S_inputs, outputs=outputs)
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    """
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            (None, None)         0                                            
    __________________________________________________________________________________________________
    embedding (Embedding)           (None, None, 128)    2560000     input_1[0][0]                    
    __________________________________________________________________________________________________
    attention (Attention)           (None, None, 128)    49152       embedding[0][0]                  
                                                                     embedding[0][0]                  
                                                                     embedding[0][0]                  
    __________________________________________________________________________________________________
    global_average_pooling1d (Globa (None, 128)          0           attention[0][0]                  
    __________________________________________________________________________________________________
    dropout (Dropout)               (None, 128)          0           global_average_pooling1d[0][0]   
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1)            129         dropout[0][0]                    
    ==================================================================================================
    Total params: 2,609,281
    Trainable params: 2,609,281
    Non-trainable params: 0
    __________________________________________________________________________________________________
    None
    """
    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=5,
              validation_data=(x_test, y_test))
    """
    24896/25000 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9696
    24928/25000 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9696
    24960/25000 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9696
    24992/25000 [============================>.] - ETA: 0s - loss: 0.0835 - acc: 0.9696
    25000/25000 [==============================] - 84s 3ms/sample - loss: 0.0835 - acc: 0.9696 - val_loss: 0.7551 - val_acc: 0.7975
    """