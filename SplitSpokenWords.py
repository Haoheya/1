#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:29:20 2021

@author: limbu
"""

#Segmenting Audio files into single word/Phrase files using silence in between spoken words

from pydub import AudioSegment
from pydub.silence import split_on_silence

sound_file = AudioSegment.from_mp3("ENSV001.mp3")
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=850,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-35
)
j = 1
for i, chunk in enumerate(audio_chunks):
    out_file = "Part"+str(j)+".mp3"
    print("exporting", out_file)
    chunk.export(out_file, format="mp3")
    j = j+1
    