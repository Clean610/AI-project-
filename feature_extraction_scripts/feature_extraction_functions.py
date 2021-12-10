#save info
import csv
import sys
from pathlib import Path

#audio 
import librosa
import librosa.display
import matplotlib.pyplot as plt

#data prep
import numpy as np
import random

#my own speech prep: voice activity detection
import feature_extraction_scripts.prep_noise as prep_data_vad_noise
from feature_extraction_scripts.errors import NoSpeechDetected, LimitTooSmall,FeatureExtractionFail

   
#load wavefile, set settings for that
def get_samps(wavefile,sr=None,high_quality=None):
    if sr is None:
        sr = 16000
    if high_quality:
        quality = "kaiser_high"
    else:
        quality = "kaiser_fast"
    y, sr = librosa.load(wavefile,sr=sr,res_type=quality) 
    
    return y, sr

#set settings for mfcc extraction
def get_mfcc(y,sr,num_mfcc=None,window_size=None, window_shift=None):
 
    if num_mfcc is None:
        num_mfcc = 40
    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
    mfccs = librosa.feature.mfcc(y,sr,n_mfcc=num_mfcc,hop_length=hop_length,n_fft=n_fft)
    mfccs = np.transpose(mfccs)
    mfccs -= (np.mean(mfccs, axis=0) + 1e-8)
    
    return mfccs


def get_domfreq(y,sr):

    frequencies, magnitudes = get_freq_mag(y,sr)
    dom_freq_index = [np.argmax(item) for item in magnitudes]
    dom_freq = np.array([frequencies[i][item] for i,item in enumerate(dom_freq_index)])
   
    
    return np.array(dom_freq)


def get_freq_mag(y,sr,window_size=None, window_shift=None):

    if window_size is None:
        n_fft = int(0.025*sr)
    else:
        n_fft = int(window_size*0.001*sr)
    if window_shift is None:
        hop_length = int(0.010*sr)
    else:
        hop_length = int(window_shift*0.001*sr)
 
    frequencies,magnitudes = librosa.piptrack(y,sr,hop_length=hop_length,n_fft=n_fft)
    frequencies = np.transpose(frequencies)
    magnitudes = np.transpose(magnitudes)
    
    return frequencies, magnitudes

#saving a lot of features in the exact shape I wanted was easiest done with .npy files. It's fast to save and fast to load.
def save_feats2npy(labels_class,dict_labels_encoded,data_filename4saving,max_num_samples,dict_class_dataset_index_list,paths_list,labels_list,feature_type,num_filters,num_feature_columns,time_step,frame_width,head_folder,limit=None,dataset_index=None):
    if dataset_index is None:
        dataset_index = 0
    #dataset_index represents train (0), val (1) or test (2) datasets

    #create empty array to fill with values
    if limit:
        max_num_samples = int(max_num_samples*limit)
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    else:
        expected_rows = max_num_samples*len(labels_class)*frame_width*time_step
    feats_matrix = np.zeros((expected_rows,num_feature_columns+1)) # +1 for the label
   
    msg = "\nFeature Extraction: Section {} of 3\nNow extracting features: {} wavefiles per class.\nWith {} classes, processing {} wavefiles.\nFeatures will be saved in the file {}.npy\n\n".format(dataset_index+1,max_num_samples,len(labels_class),len(labels_class)*max_num_samples,data_filename4saving)
    print(msg)
    
    
    row = 0

    completed = False

    
    try:
        if expected_rows < 1*frame_width*time_step:
          
            raise LimitTooSmall("\nIncrease Limit: The limit at '{}' is too small.".upper().format(limit))
        
      
        paths_labels_list_dataset = []
        for i, label in enumerate(labels_class):
     
            train_val_test_index_list = dict_class_dataset_index_list[label]
            
            for k in train_val_test_index_list[dataset_index]:
                paths_labels_list_dataset.append((paths_list[k],labels_list[k]))
        
      
        random.shuffle(paths_labels_list_dataset)
        
        for wav_label in paths_labels_list_dataset:

            if row >= feats_matrix.shape[0]:
                break
            else:
                wav_curr = wav_label[0]
                label_curr = wav_label[1]
                #integer encode the label:
                label_encoded = dict_labels_encoded[label_curr]
                
                #function below basically extracts the features and makes sure each sample's features are the same size: they are cut short
                #if too long and zero padded if too short
                feats = coll_feats_manage_timestep(time_step,frame_width,wav_curr,feature_type,num_filters,num_feature_columns,head_folder)
                
                #add label column - need label to stay with the features!
                label_col = np.full((feats.shape[0],1),label_encoded)
                feats = np.concatenate((feats,label_col),axis=1)
                
                #fill the matrix with the features just collected
                feats_matrix[row:row+feats.shape[0]] = feats
                
                #actualize the row for the next set of features to fill it with
                row += feats.shape[0]
                
                #print on screen the progress
                progress = row / expected_rows * 100
                sys.stdout.write("\r%d%% through current section" % progress)
                sys.stdout.flush()
        print("\nRow reached: {}\nSize of matrix: {}\n".format(row,feats_matrix.shape))
        completed = True
    
    except LimitTooSmall as e:
        print(e)

    finally:
        np.save(data_filename4saving+".npy",feats_matrix)
        
    return completed


#this function feeds variables on to the feature extraction function 'get_feats'

def coll_feats_manage_timestep(time_step,frame_width,wav,feature_type,num_filters,num_feature_columns,head_folder):
    feats = get_feats(wav,feature_type,num_filters,num_feature_columns,head_folder)
    max_len = frame_width*time_step
    if feats.shape[0] < max_len:
        diff = max_len - feats.shape[0]
        feats = np.concatenate((feats,np.zeros((diff,feats.shape[1]))),axis=0)
    else:
        feats = feats[:max_len,:]
    
    return feats


def get_feats(wavefile,feature_type,num_features,num_feature_columns,head_folder,delta=False,dom_freq=False,noise_wavefile = None,vad = False):
    y, sr = get_samps(wavefile)

    if vad:
        try:
            y, speech = prep_data_vad_noise.get_speech_samples(y,sr)
            if speech:
                pass
            else:
                raise NoSpeechDetected("\n!!! FYI: No speech was detected in file: {} !!!\n".format(wavefile))
        except NoSpeechDetected as e:
            print("\n{}".format(e))
            filename = '{}/no_speech_detected.csv'.format(head_folder)
            with open(filename,'a') as f:
                w = csv.writer(f)
                w.writerow([wavefile])

    extracted = []
    if "mfcc" in feature_type.lower():
        extracted.append("mfcc")
        features = get_mfcc(y,sr,num_mfcc=num_features)
        if delta:
            delta, delta_delta = get_change_acceleration_rate(features)
            features = np.concatenate((features,delta,delta_delta),axis=1)
    
    if dom_freq:
        dom_freq = get_domfreq(y,sr)
        dom_freq = dom_freq.reshape((dom_freq.shape+(1,)))
        features = np.concatenate((features,dom_freq),axis=1)
    if features.shape[1] != num_feature_columns: 
        raise FeatureExtractionFail("The file '{}' results in the incorrect  number of columns (should be {} columns): shape {}".format(wavefile,num_feature_columns,features.shape))
    
    return features

def unique_path(directory, name_pattern):
    counter = 0
    while True:
        counter += 1
        path = directory / name_pattern.format(counter)
        if not path.exists():
            return path

