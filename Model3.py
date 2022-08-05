#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:32:17 2021

"""

#Model 3 : Introduction of BiDirectional LSTM + extra Decoder LSTM + MultiHead Attention

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, RepeatVector, Input, Concatenate, TimeDistributed
from tensorflow.keras import layers
from keras.layers import MultiHeadAttention
import pickle

Eng_Pickle = 'Zh97.pckl'                                #Load English .pckl file
Swe_Pickle = 'Tibet97.pckl'                                #Load Swedish .pckl file

f = open(Eng_Pickle, 'rb')
train = pickle.load(f)
f.close()
g = open(Swe_Pickle, 'rb')
target = pickle.load(g)
g.close()

timesteps = 500
features = 128
latent_dim = 128

dropout_rate = 0.2

inputs = Input(shape=(timesteps, features))

#Initial state and cell state of the final BLSTM layer as initial states to Decoder(Done)

#Encoder
masked_encoder_inputs = layers.Masking(mask_value = 0.0)(inputs)
encoder_dropout = (TimeDistributed(Dropout(rate = dropout_rate)))(masked_encoder_inputs)

encoder_out1,f_h1, f_c1, b_h1, b_c1 = Bidirectional(LSTM(latent_dim, return_sequences = True,return_state=True))(encoder_dropout)

encoder_out2,f_h2, f_c2, b_h2, b_c2 = Bidirectional(LSTM(latent_dim, return_sequences = True, return_state=True,dropout=0.15))(encoder_out1)

encoder_out3,f_h3, f_c3, b_h3, b_c3 = Bidirectional(LSTM(latent_dim, return_sequences = True, return_state=True,dropout= 0.15))(encoder_out2)
state_h3 = Concatenate()([f_h3, b_h3])
state_c3 = Concatenate()([f_c3, b_c3])

encoder_states = [state_h3, state_c3]

#Multi-Head Attention
att_out = MultiHeadAttention(num_heads=4, key_dim=128)(encoder_out3,encoder_out2)

#Decoder
decoder_input1 = (att_out)
decoder_lstm1 = LSTM(latent_dim*2, return_state=True,return_sequences=True)
decoder_dropout1 = (TimeDistributed(Dropout(rate = dropout_rate)))(decoder_input1)
decoder_out1, dec_h, dec_c = decoder_lstm1(decoder_dropout1, initial_state = encoder_states)

decoder_states = [dec_h,dec_c]

decoder_input2 = (decoder_out1)
decoder_lstm2 = LSTM(latent_dim*2, return_state=True,return_sequences=True)
decoder_dropout2 = (TimeDistributed(Dropout(rate = dropout_rate)))(decoder_input2)
decoder_out2, _, _ = decoder_lstm2(decoder_dropout2, initial_state = decoder_states)

decoder_outputs = TimeDistributed(Dense(features, activation='relu'))(decoder_out2)

model = Model(inputs, decoder_outputs)

#Plot Model FlowChart
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model3.png', show_shapes=True, show_layer_names=True)

# Compile Model
model.compile(optimizer='adam', loss='mean_squared_error')

# Run Training
model.fit(train, target,
          batch_size=64,
          epochs=1000,
          validation_split=0.2)

#Save Model Weights
model.save('s2sIII.h5')


