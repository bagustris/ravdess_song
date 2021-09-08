# ravdess_song
Song emotion recognition on RAVDESS song dataset   

A simple tutorial/demonstration on deep learning application for emotion recognition from song data.  

## Tested on:  
Tensorflow:  2.5 and 2.6  
Librosa: 8.1  

## One Click
There are two ways to simulate, either by clone this repo or run it on Google Colab.  
For Google Colab, [click here](https://colab.research.google.com/github/bagustris/ravdess_song/blob/main/ravdess_song_sd.ipynb) (You must have a google account!).
You can also clicks in .ipynb file above and open it colab.

## Dataset 
The dataset is included in this repo. This is a part of RAVDESS [1] dataset with song data only.
Since it has a license "CC BY-NC-SA 4.0", we can provide it here.

## Dataset partition
*Speaker-dependent:*    
Train: 910 files / 90 %   
Test: 102 files / 10 %    
*Speaker-independent:*  
Train: 20 speakers (1-20) / 836 files /  83 %  
Test: 4 speakers (21-24) / 176 files / 17 %   

## What will you learn
In this tutorial you will learn:  
1. How to extract acoustic features  
2. How to build classifier: NN/MLP/FCN/Dense, LSTM, CNN  
3. How to split dataset into training and test  
4. How to make speaker-independent partition  

## Accuracy result  
*Speaker-independet:* 93.%  
*Speaker-independet:* 74.%  

## Next
Life after this (based on problem you're facing):    
1. Try to extract other acoustic features
2. Try other classifiers
3. Try to other datasets

### If you encounter a problem, submit an [issue](https://github.com/bagustris/ravdess_song/issues)
Please cite the last reference is you take a benefit of this repo.

## Reference:  
[1] S. R. Livingstone and F. A. Russo, “The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS),” PLoS One, pp. 1–35, 2018, doi: 10.5281/zenodo.1188976.  
[2] MLP: http://www.sainshack.com/2017/08/10/jaringan-syaraf-tiruan-dan-implementasinya-dalam-bahasa-r-1/  
[3] LSTM: http://colah.github.io/posts/2015-08-Understanding-LSTMs/  
[4] CNN: http://bagustris.blogspot.com/2018/08/deep-learning-cnn-konvolusi-dan-pooling.html  
[5] B. T. Atmaja and M. Akagi, “On The Differences Between Song and Speech Emotion Recognition: Effect of Feature Sets, Feature Types, and Classifiers,” in 2020 IEEE REGION 10 CONFERENCE (TENCON), 2020, pp. 968–972.
