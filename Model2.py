#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:31:51 2021

@author: limbu
"""

#Model 2 Introduction of Dropouts and BiDirectional LSTMs

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, RepeatVector, Input, Concatenate, TimeDistributed
from tensorflow.keras import layers
import pickle

Eng_Pickle = 'Eng.pckl'                                #Load English .pckl file
Swe_Pickle = 'Swe.pckl'                                #Load Swedish .pckl file

f = open(Eng_Pickle, 'rb')
train = pickle.load(f)
f.close()
g = open(Swe_Pickle, 'rb')
target = pickle.load(g)
g.close()

timesteps = 200
features = 128
latent_dim = 128
#Dropout Rate for both encoder and decoder inputs
dropout_rate = 0.2

inputs = Input(shape=(timesteps, features))

#Encoder
#Use Masking to make sure all the different inputs and outputs have same length for different sentences and translations
masked_encoder_inputs = layers.Masking(inputs)
encoder_dropout = (TimeDistributed(Dropout(rate = dropout_rate)))(masked_encoder_inputs)

encoder_out1,f_h1, f_c1, b_h1, b_c1 = Bidirectional(LSTM(latent_dim, return_sequences = True,return_state=True))(encoder_dropout)
state_h1 = Concatenate()([f_h1, b_h1])
state_c1 = Concatenate()([f_c1, b_c1])

encoder_out2,f_h2, f_c2, b_h2, b_c2 = Bidirectional(LSTM(latent_dim, return_sequences = True, return_state=True,dropout=0.3))(encoder_out1)
state_h2 = Concatenate()([f_h2, b_h2])
state_c2 = Concatenate()([f_c2, b_c2])

#Concatenate forward and backward for both hidden and cell states
state_h = Concatenate()([state_h1,state_h2])
state_c = Concatenate()([state_c1,state_c2])

encoder_states = [state_h, state_c]

#Decoder
decoder_inputs = (encoder_out3)
decoder_lstm = LSTM(latent_dim*4, return_state=True,return_sequences=True)
decoder_dropout = (TimeDistributed(Dropout(rate = dropout_rate)))(decoder_inputs)
decoder_outputs, _, _ = decoder_lstm(decoder_dropout, initial_state = encoder_states)
decoder_outputs = TimeDistributed(Dense(features, activation='relu'))(decoder_outputs)

model = Model(inputs, decoder_outputs)

#Plot Model FlowChart
from keras.utils import plot_model
plot_model(model, to_file='model2.png', show_shapes=True, show_layer_names=True)

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Run Training
model.fit(train, target,
          batch_size=64,
          epochs=1000,
          validation_split=0.2)

#Save Model Weights
model.save('s2sII.h5')

#Prediction model for training samples
y_predS = model.predict(train)

#Predicted Audio
start = time.time()
print("Timing starts now")
mel = np.abs(np.exp(y_predS[0].T))
res = librosa.feature.inverse.mel_to_audio(mel)
res = (np.iinfo(np.int32).max * (res/np.abs(res).max())).astype(np.int32)
end = time.time() 
print(end - start)

import IPython.display as ipd
ipd.Audio(res,rate = sr)
