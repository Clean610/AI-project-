import sys
import matplotlib as plt 
import pyaudio
import numpy as np
import matplotlib.mlab as mlab
from queue import Queue
from pydub import AudioSegment
from pydub.playback import play
from keras.models import load_model
import feature_extraction_scripts.feature_extraction_functions as featfun
import csv
import time
import os
import datetime
import soundfile as sf
import librosa
import feature_extraction_scripts.prep_noise as pn


def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string
    

def main_detect_trigger(project_head_folder,model_name):
    
    head_folder_beg = "./ml_speech_projects/"
    head_folder_curr_project = head_folder_beg+project_head_folder
    
    #load the information related to features and model of interest
    features_info_path = head_folder_curr_project+"/features_log.csv"
    encoded_label_path = head_folder_curr_project+"/labels_encoded.csv"
    model_path =  head_folder_curr_project+"/models/{}.h5".format(model_name)
    model_log_path = head_folder_curr_project+"/model_logs/{}.csv".format(model_name)
    
    #find out the settings for feature extraction
    with open(features_info_path, mode='r') as infile:
        reader = csv.reader(infile)            
        feats_dict = {rows[0]:rows[1] for rows in reader}
    feature_type = feats_dict['features']
    num_filters = int(feats_dict['num original features'])
    num_feature_columns = int(feats_dict['num total features'])
    delta = str2bool(feats_dict["delta"])
    dom_freq = str2bool(feats_dict["dominant frequency"])
    vad = str2bool(feats_dict["beginning silence removal"])
    timesteps = int(feats_dict['timesteps'])
    context_window = int(feats_dict['context window'])
    frame_width = context_window*2+1
      
    #prepare the dictionary to find out the assigned label 
    with open(encoded_label_path, mode='r') as infile:
        reader = csv.reader(infile)
        label_for_dict = []  
        for rows in reader:
            label_for_dict.extend(rows)
            print(label_for_dict)
            print(label_for_dict[1])
        # dict_labels_encoded = {}
        dict_labels_encoded = {label_for_dict[0]:label_for_dict[1] , label_for_dict[2]:label_for_dict[3], label_for_dict[4]:label_for_dict[5], label_for_dict[6]:label_for_dict[7], label_for_dict[8]:label_for_dict[9] }
        # for i in range(label_for_dict):
        #     dict_labels = {label_for_dict[1],label_for_dict[2]}
        #     dict_labels_encoded.append(dict_labels)
        print(dict_labels_encoded)
    
    print("\nAvailable labels:")
    for key, value in dict_labels_encoded.items():
        print(value)
        
    features = featfun.coll_feats_manage_timestep(timesteps,frame_width,feature_type,num_filters,num_feature_columns,delta=delta,dom_freq=dom_freq,vad=vad)
     
    with open(model_log_path, mode='r') as infile:
        reader = csv.reader(infile)            
        dict_model_settings = {rows[0]:rows[1] for rows in reader}
        
    model_type = dict_model_settings["model type"]
    activation_output = dict_model_settings["activation output"]
    X = features
    if model_type == "lstm":
        X = X.reshape((timesteps,frame_width,X.shape[1]))
    elif model_type == "cnn":
        X = X.reshape((X.shape[0],X.shape[1],1))
        X = X.reshape((1,)+X.shape)
    elif model_type == "cnnlstm":
        X = X.reshape((timesteps,frame_width,X.shape[1],1))
        X = X.reshape((1,)+X.shape)        

    model = load_model(model_path)
    
    prediction = model.predict(X)
    pred = str(np.argmax(prediction[0]))

    
    label = dict_labels_encoded[pred]
    print("Label without noise reduction: {}".format(label))
    
    return prediction.reshape()

if __name__=="__main__":
    
    project_head_folder = "mfcc13_models_2021y11m16d17h8m56s"
    model_name = "CNNLSTM_speech_commands001"
    
    main_detect_trigger(project_head_folder,model_name)
  


def has_new_triggerword(prediction, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.
    Argument:d
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive
    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = prediction > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


chunk_duration = 2 # Each read length in seconds from mic.
fs = 16000 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)



# def get_spectrogram(data):
#     """
#     Function to compute a spectrogram.
    
#     Argument:
#     predictions -- one channel / dual channel audio data as numpy array

#     Returns:
#     pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
#     """
#     nfft = 200 # Length of each window segment
#     fs = 16000 # Sampling frequencies
#     noverlap = 120 # Overlap between windows
#     nchannels = data.ndim
#     if nchannels == 1:
#         pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap = noverlap)
#     # elif nchannels == 2:
#     #     pxx, _, _ = mlab.specgram(data[:,0], nfft, fs, noverlap = noverlap)
#     print("pxx==>",pxx)   
#     return pxx


# def plt_spectrogram(data):
#     """
#     Function to compute and plot a spectrogram.
    
#     Argument:
#     predictions -- one channel / dual channel audio data as numpy array

#     Returns:
#     pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
#     """
#     nfft = 200 # Length of each window segment
#     fs = 16000 # Sampling frequencies
#     noverlap = 120 # Overlap between windows
#     nchannels = data.ndim
#     if nchannels == 1:
#         pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap = noverlap)
#     elif nchannels == 2:
#         pxx, _, _, _ = plt.specgram(data[:,0], nfft, fs, noverlap = noverlap)
#     return pxx


        


def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream

     

# Queue to communiate between the audio callback and main thread
q = Queue()

run = True

silence_threshold = 100

# Run the demo for a timeout seconds
timeout = time.time() + 0.5*60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')

def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold    
    if time.time() > timeout:
        run = False        
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data,data0)    
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)

stream = get_audio_input_stream(callback)
stream.start_stream()


try:
    while run:
        data = q.get()
        mfcc = main_detect_trigger(data)
        preds = mfcc
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        if new_trigger:
            sys.stdout.write('1')
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False
        
stream.stop_stream()
stream.close()

