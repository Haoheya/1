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
#######Keywords: Encoder-Decoder, Sequence to Sequence Modelling, LSTM Neural
Networks
