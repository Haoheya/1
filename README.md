# Direct Speech to Speech Translation Using Machine learning
A proof of concept model to translate languages directly from source to target audio using Machine Learning. 

## Abstract
Nowadays, most of the speech to speech translation applications and services
use a three step process. First step is the speech to text translation using speech
recognition. This is followed by text to text language translation and finally the text
is synthesized into speech. As the availability of data and computing power improved,
each of these individual steps advanced over time. Although, the progress
was significant, there was always error associated with the first translation step in
terms of various factors such as tone recognition of the speech, accent etc. The error
further propagated and quite often deteriorated as itwent down the translation steps.
This gave birth to an ongoing budding research in direct speech to speech translation
without relying on text translations. This project is inspired from Google’s
’Translatotron : An End-to-End Speech-to-Speech translation model’. In line with
the ’Translatotron’ model this thesis makes use of a simpler Sequence-to-Sequence
(STS) encoder-decoder LSTM network using spectrograms as input to examine the
possibility of direct language translations in audio form. Although the final results
have inconsistencies and are not as efficient as the traditional speech to speech translation
techniques which heavily rely on text translations, they serve as a promising
platform for further research.
###### Keywords: Encoder-Decoder, Sequence to Sequence Modelling, LSTM Neural Networks

## Project Description
This model takes a text corpus translated from Swedish to English from an online source (https://opus.nlpl.eu/) as the training dataset. This Corpus contains two text files one containing Swedish sentences and the other the translated English sentences line by line. It is important to consider at this point that both the text files should maintain proper sequence in terms of the Swedish and the translated English sentence line by line i.e. line number 8 in the Eng text file should contain the translation of line number 8 in Sv file. So on and so forth. In case this is difficult to achieve an alternative is to use Google Translation API on the Sv text file line by line.

We then use Google TTS Wavenet API to synthesize this text to speech. Kindly note, this would not be 'cheating' the main objective of this project as the Google API has not been used to perform the actual translation process. This gives us the training set of utterance pairs from Swedish to English. 

We build three models in the order of increasing 'clarity' of results.

This proof of concept model translates the Swedish utterance that is present in the training dataset. Although this may seem obvious, the normal ML models do not work properly to this end at all.

Note : The project also uses another online source for dataset - https://www.101languages.net/swedish/. This website provides a set of 100 files which has Swedish to English translated in human voice. In my project analysis this source seems to produce better results.

## Structure of the Project Files and How to Use it
1. Use the 'Line2Text.py' file by placing the aforementioned downloaded corpus in the Source file location for Swedish and English accordingly.
2. After the corpus has been broken in text files containing single sentences, use TTSWavenet.py to synthesize it into speech. Please note that for this step you will have to create a project in Google Cloud (GCP). The coversions are free of cost as of 28th June 2021.
3. In this step we create a 3D matrix of the spectrogram for both the datasets using 'GenerateData.py'. For this to work kindly change "C:\src\Thesis\Data\Tat\Speech\Test\Eng\En" to the location where you synthesized the utterance pairs in the previous step. The 3D matrices will be stored as '.pckl' files for both Eng and Swe.
4. Now you use any (or all) of the three models by loading the '.pckl' file and run the code. All the three models should generate the predicted audio for the first training Swedish utterance at the end.

Tips:
To run the training model, one can use Kaggle Kernel that provides free powerful GPU.
