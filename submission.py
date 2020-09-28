import librosa as lr
from librosa.core import stft
from librosa.core import amplitude_to_db
from librosa.display import specshow
from librosa.feature import melspectrogram
import soundfile as sf
import wave
from pylab import *


import pandas as pd
import numpy as np
import sys, os
from random import randint
from glob import glob

import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


SOUNDFILE = './rooster_competition.wav'
RATE = 44100
FRAME = 512
N_MFCC = 15
STRIDE = 250
WINDOW_LENGTH = 2500
THRESHOLD = 1

def print_file_info(file):
    print(sf.info(file))

def plot_waveform(file):
    audio, sr = lr.load(SOUNDFILE, sr=RATE)
    time = np.arange(len(audio)) / sr

    fig = plt.figure(figsize=(9,3))
    plt.plot(time, audio)
    plt.xlim((0,len(audio)/sr))
    plt.title('Waveform of the Audio File')
    plt.xlabel('Time in s')
    plt.ylabel('Sound Amplitude')
    plt.tight_layout()
    plt.show()
    # fig.savefig('images/competition_waveform')

def plot_envelope(file):
    audio, sr = lr.load(SOUNDFILE, sr=RATE)
    time = np.arange(len(audio)) / sr
    df = pd.DataFrame(list(zip(time,audio)),columns=['time','amplitude'])
    rect = df['amplitude'].apply(np.abs)
    envelope = rect.rolling(10000).mean()

    fig = plt.figure(figsize=(9,3))
    plt.xlim((0,len(audio)/sr))
    plt.xlabel('Time in s')
    plt.ylabel('Average Sound Amplitude')
    plt.plot(time,envelope)
    plt.title('Rolling Average of the absolute Sound Amplitude')
    plt.tight_layout()
    plt.show()
    # fig.savefig('images/competition_rolling_average')

def plot_spectrogram(file):
    audio, sr = lr.load(SOUNDFILE, sr=RATE)
    time = np.arange(len(audio)) / sr
    spec = stft(audio, hop_length=FRAME, n_fft=2**7)
    spec_db = amplitude_to_db(np.abs(spec))

    fig, ax = plt.subplots(figsize=(9,3))
    specshow(spec_db, sr=sr, x_axis='time', y_axis='hz', hop_length=FRAME, ax=ax, cmap='magma')
    fig.suptitle('Spectrogram of the Recording')
    ax.set_ylabel('Frequency in Hz')
    ax.set_xlabel('Time in min:s')
    plt.tight_layout()
    plt.show()
    # fig.savefig('images/competition_spectrogram')

def plot_esc50_spectrograms():
    esc50dir = './dataset/ESC-50-master/'
    esc50audio = esc50dir + 'audio/'
    esc50meta = esc50dir + 'meta/'
    esc50 = glob(esc50audio + '*.wav')
    esc50 = [s[len(esc50audio):] for s in esc50]

    meta = pd.read_csv(esc50meta + 'esc50.csv')
    roosters = list(meta[meta['category'] == 'rooster']['filename'])
    breathing = list(meta[meta['category'] == 'crow']['filename'])
    hens = list(meta[meta['category'] == 'hen']['filename'])

    fig, axs = plt.subplots(3,2,figsize=(10,8),sharex=True)
    fig.suptitle('Comparison between different Classes')
    i = randint(0,meta.shape[1])
    files = [roosters[i],breathing[i],hens[i]]
    names = ['a Rooster','a Crow','a Hen']
    for j,f in enumerate(files):
        a,f = lr.load(esc50audio+f, sr=RATE)
        t = np.arange(0, len(a)) / f

        axs[j][0].plot(t,a)
        axs[j][0].set_title('Waveform of '+names[j])
        axs[j][0].set_xlabel('Time in s')
        axs[j][0].set_ylabel('Amplitude')

        spec = stft(a, hop_length=FRAME, n_fft=2**7)
        spec_db = amplitude_to_db(np.abs(spec))
        specshow(spec_db, sr=f, x_axis='time', y_axis='hz', hop_length=FRAME, ax = axs[j][1], cmap='magma')

        axs[j][1].set_title('Spectrogram of '+names[j])
        axs[j][1].set_xlabel('Time in s')
        axs[j][1].set_ylabel('Frequency in Hz')

    plt.tight_layout()
    plt.show()
    # fig.savefig('images/esc_50_class_comparison', dpi=300)

def plot_channel_differences(file):

    waveFile = wave.open(file,'rb')
    nframes = waveFile.getnframes()
    fr = waveFile.getframerate()
    wavFrames = waveFile.readframes(nframes)
    waveFile.close()

    amplitudes = np.frombuffer(wavFrames, dtype=np.int16)
    data = amplitudes.reshape(-1,2)
    df = pd.DataFrame(data,columns=['amplitude_left','amplitude_right'])
    df['time'] = np.arange(nframes)/fr
    df['left_abs'] = df['amplitude_left'].apply(abs)
    df['right_abs'] = - df['amplitude_right'].apply(abs)
    df['difference'] = df['left_abs'] + df['right_abs']
    df['left_abs_avg'] = df['left_abs'].rolling(50000).mean()
    df['right_abs_avg'] = df['right_abs'].rolling(50000).mean()
    df['difference_avg'] = df['left_abs_avg'] + df['right_abs_avg']

    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(list(df['time']),np.zeros(df.shape[0]),c='k')
    df.plot(x='time',y='left_abs_avg',ax=ax, c='C0', label='left Channel')
    df.plot(x='time',y='right_abs_avg',ax=ax, c='C2', label='right Channel')
    df.plot(x='time',y='difference_avg',ax=ax, c='C3', label='Difference')
    ax.set_title('Comparison between the left and right Channel')
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    # fig.savefig('images/competition_channel_comparison', dpi=300)

def create_sound_windows(file):
    subtype = 'PCM_16'
    duration = 5

    audio, sr = lr.load(file, sr=RATE)
    print('Will create',len(audio)//(duration*sr),'subfiles,',duration,'seconds each')
    for j in np.arange(len(audio)//(duration*sr)):
        a = audio[j*duration*sr:(j+1)*duration*sr]
        name = 'window_'+str(j)+'.wav' 
        p = os.path.abspath('dataset/search-windows/'+name)
        sf.write(file=p, data=a, samplerate = RATE, subtype=subtype)
        print('created',name)

def create_noise_sounds():
    subtype = 'PCM_16'
    duration = 5
    birdsFile = 'dataset/birds.wav'
    marketFile = 'dataset/market.wav'
    restaurantFile = 'dataset/restaurant.wav'

    names = ['birds','market','restaurant']
    files = [birdsFile, marketFile, restaurantFile]

    for i,f in enumerate(files):
        audio, sr = lr.load(f, sr=RATE)
        print('Will create',len(audio)//(duration*sr),'subfiles,',duration,'seconds each')
        for j in np.arange(len(audio)//(duration*sr)):
            a = audio[j*duration*sr:(j+1)*duration*sr]
            name = names[i]+'_'+str(j)+'.wav' 
            p = os.path.abspath('dataset/noise/'+name)
            sf.write(file=p, data=a, samplerate = RATE, subtype=subtype)
            print('created',name)

def get_mfcc_df(files, n_mfcc = N_MFCC):
    df = pd.DataFrame()

    for f in files:
        dfTemp = pd.DataFrame()
        audio, sr = lr.load(f, sr=RATE)
        
        melspec = lr.feature.melspectrogram(audio, sr=RATE, hop_length=FRAME)
        dbAmplitude = lr.amplitude_to_db(melspec)
        mfcc = lr.feature.mfcc(S=dbAmplitude, n_mfcc=n_mfcc).transpose()
        mfcct = mfcc.T
        for n in np.arange(mfcct.shape[0]):
            dfTemp['mean_mfcc_'+str(n)] = [np.mean(mfcct[n])]
            dfTemp['std_mfcc_'+str(n)] = [np.std(mfcct[n])]
            dfTemp['median_mfcc_'+str(n)] = [np.median(mfcct[n])]
            dfTemp['max_mfcc_'+str(n)] = [np.max(mfcct[n])]

        df = df.append(dfTemp)
    return df

def plot_mfcc_clusters():
    esc50records = glob('./dataset/ESC-50-master/audio/*.wav')
    roosterFiles = [i for i in esc50records if int(i.split('-')[-1].split('.')[0]) == 1]
    noiseFiles = glob('./dataset/noise/*.wav')
    windowFiles = glob('./dataset/search-windows/*.wav')

    dfRooster = get_mfcc_df(roosterFiles, n_mfcc=13)
    dfNoise = get_mfcc_df(noiseFiles, n_mfcc=13)
    dfComp = get_mfcc_df(windowFiles, n_mfcc=13)

    fig,ax = plt.subplots()
    dim1 = 'mean_mfcc_0'
    dim2 = 'mean_mfcc_2'
    dfRooster.plot.scatter(x=dim1, y=dim2, ax=ax, c='C0', label='Roosters')
    dfNoise.plot.scatter(x=dim1, y=dim2, ax=ax, c='C2', label='Noise')
    dfComp.plot.scatter(x=dim1, y=dim2, ax=ax, c='C3', label='Recording')

    fig.suptitle('Scatter Plot of different Clusters')
    plt.tight_layout()
    plt.show()
    # fig.savefig('images/mfcc-clusters', dpi=300)

def train():
    # Collect all necessary sound files for training
    esc50records = glob('./dataset/ESC-50-master/audio/*.wav')
    roosterFiles = [i for i in esc50records if int(i.split('-')[-1].split('.')[0]) == 1]
    noiseFiles = glob('./dataset/noise/*.wav')

    # Extract the necessary features
    n_mfcc = N_MFCC
    dfRooster = get_mfcc_df(roosterFiles, n_mfcc=n_mfcc)
    dfNoise = get_mfcc_df(noiseFiles, n_mfcc=n_mfcc)
    dfRooster['isRooster'] = np.ones(dfRooster.shape[0])
    dfNoise['isRooster'] = np.zeros(dfNoise.shape[0])
    df = dfRooster.append(dfNoise)

    # Build the training and testing data
    y = df['isRooster'].values
    X = df.drop(['isRooster'], axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)

    # Set up the training pipeline
    steps = [('scaler', StandardScaler())]
    # Choose one of the classification models:
    # steps += [('knn', KNeighborsClassifier(n_neighbors=2))]
    # steps += [('logreg', LogisticRegression())]
    # steps += [('SVM', SVC())]
    steps += [('tree',DecisionTreeClassifier(max_depth=10, random_state=42))]
    pipeline = Pipeline(steps)

    # Training
    pipeline.fit(X_train,y_train)
    testScore = pipeline.score(X_test, y_test)
    #print('Test score:',testScore)

    # Generate the confusion matrix and classification report
    y_pred = pipeline.predict(X_test)
    #print('Confusion matrix:',confusion_matrix(y_test,y_pred),sep='\n')
    #print(classification_report(y_test,y_pred))
    return pipeline, testScore

def get_window_df(file, wl = WINDOW_LENGTH, stride = STRIDE):
    audio, sr = lr.load(file, sr=RATE)
    n_mfcc = N_MFCC
    df = pd.DataFrame()

    # calculate window size based on milliseconds
    windowLength = sr*wl//1000
    # num_w = len(audio)//windowLength
    # frameSize = windowLength
    #for i in range(nw):
    j = k = 0
    last = len(audio)
    while(j < last):
        # start and end point of the window
        i = sr*k*stride//1000
        j = i + windowLength
        # print('from {} to {}'.format(i/sr,j/sr))
        dfTemp = pd.DataFrame()#columns = cols)
        a = audio[i:j]#[i*windowLength:(i+1)*windowLength]
        msp = lr.feature.melspectrogram(a, sr=RATE, hop_length=FRAME)
        dbAmplitude = lr.amplitude_to_db(msp)
        mfcc = lr.feature.mfcc(S=dbAmplitude, n_mfcc=n_mfcc).transpose()
        mfcct = mfcc.T
        for n in np.arange(mfcct.shape[0]):
            dfTemp['mean_mfcc_'+str(n)] = [np.mean(mfcct[n])]
            dfTemp['std_mfcc_'+str(n)] = [np.std(mfcct[n])]
            dfTemp['median_mfcc_'+str(n)] = [np.median(mfcct[n])]
            dfTemp['max_mfcc_'+str(n)] = [np.max(mfcct[n])]
        dfTemp['start'] = int(i/sr*1000)
        dfTemp['end'] = int(j/sr*1000)

        df = df.append(dfTemp)
        k += 1
        

    return df

def get_durations(array):
    count = 0
    durations = []
    for _,val in enumerate(array):
        if val >0:
            count +=1
        else:
            if count > 0:
                durations.append(count)
            count = 0
    return durations

def print_results(array,stride):
    durations = get_durations(array)
    dfResults = pd.DataFrame()
    for i,val in enumerate(durations):
        dfRoosterEntry = pd.DataFrame()
        dfRoosterEntry['name'] = [i+1]
        dfRoosterEntry['duration'] = [val*stride]
        dfResults = dfResults.append(dfRoosterEntry)
    print('Total Number of Roosters:',len(durations),end='\n\n')
    print('Crow Durations:')
    for i in range(dfResults.shape[0]):
        print('{}:{}'.format(dfResults['name'].values[i],dfResults['duration'].values[i]))
        
    print('')
    print('Rooster Ranking:')
    for i,val in enumerate(dfResults.sort_values(by='duration',ascending=False)['name'].values):
        print('{}:{}'.format(i+1,val))

def get_results(pipeline, testScore):
    windowLength = WINDOW_LENGTH
    stride = STRIDE
    
    dfWindow = get_window_df(SOUNDFILE, wl = windowLength, stride = stride)
    X_res = dfWindow.drop(['start','end'],axis=1).values
    results = pipeline.predict(X_res)
    
    dfWindow['start'] = [stride*i for i in range(len(results))]
    dfWindow['end'] = [stride*i+windowLength for i in range(len(results))]
    dfWindow['prediction'] = results
    
    dfPositive = dfWindow[dfWindow['prediction']==1].reset_index()
    
    last = dfWindow['end'].values[-1]//stride
    
    dfPosCount = pd.DataFrame()
    dfPosCount['section']=[i*stride/2 for i in range(1,2*last) if i%2 == 1]
    dfPosCount['count'] = np.zeros(last)


    for i in range(dfPositive.shape[0]):
        s = dfPositive['start'][i]
        e = dfPositive['end'][i]
        for i in range(s//stride,e//stride):
            if i < dfPosCount.shape[0]:
                dfPosCount['count'][i] = int(dfPosCount['count'][i]+1)


    audio, sr = lr.load(SOUNDFILE,sr=RATE)
    time = np.arange(0, len(audio)) / sr
    df = pd.DataFrame(list(zip(time,audio)),columns=['time','amplitude'])
    rect = df['amplitude'].apply(np.abs)
    envelope = rect.rolling(10000).mean()*3
    
    timePos = np.arange(1,dfPosCount.shape[0]+1)*stride-stride/2
    timePos = timePos/1000
    # timePos = np.linspace(0,time[-1],dfPosCount.shape[0])
    print(timePos[-1])

    # break down the result counts into binary results based on a threshold
    threshold = THRESHOLD
    counts = dfPosCount['count'].values
    classified = []
    for i,val in enumerate(counts):
        if val > threshold - 1:
            classified.append(1)
        else:
            classified.append(0)

    print_results(classified,stride)
    
    fig, ax = plt.subplots(figsize=(9,3))
    fig.suptitle('Overlayed Classification (w={}ms, stride={}ms, #MFCC={})\nTrained with decision tree, test_acc={}'.format(windowLength,stride,N_MFCC,testScore))
    ax.set_xlim((0,len(audio)/sr))
    ax.plot(time,envelope,c='C0',zorder=1)
    ax.set_xlabel('Time in s')
    ax.set_ylabel('Sound Envelope')
    
    ax2 = ax.twinx()
    ax2.set_ylim([-1,7])
    ax2.set_yticks([0,1])

    ax2.scatter(timePos,classified,marker='.',s=4,c='C3',zorder=2)
    # ax2.scatter(timePos,counts,s=1,marker='x',c='C4',zorder=3)
    plt.tight_layout()
    fig.savefig('images/results_tree10_w{}_s{}_mfcc{}_thr{}_all.png'.format(windowLength,stride,N_MFCC,THRESHOLD))
    plt.show()


# print_file_info(SOUNDFILE)
# plot_waveform(SOUNDFILE)
# plot_envelope(SOUNDFILE)
# plot_spectrogram(SOUNDFILE)
# plot_esc50_spectrograms()
# plot_channel_differences(SOUNDFILE)
# create_sound_windows(SOUNDFILE)
# create_noise_sounds()
# plot_mfcc_clusters()
pipeline, testScore = train()
get_results(pipeline, testScore)


# tmp = [0,1,1,1,0,0,1,0,0,1,1,0]
# print(get_durations(tmp))