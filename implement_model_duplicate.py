import os
import csv
import datetime
import numpy as np
from numpy.lib.financial import rate
import sounddevice as sd
import soundfile as sf
import librosa
from keras.models import load_model
import time as tm
import sys
import feature_extraction_scripts.feature_extraction_functions as featfun
import feature_extraction_scripts.prep_noise as pn
import pyaudio
from scipy.io.wavfile import write
import threading
from array import array
from queue import Queue, Full

import pyaudio






def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day,time.hour,time.minute,time.second)
    return(time_str)


def listen(stopped, q):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024,
    )
    sr=rate
    while True:
        if stopped.wait(timeout=0):
            break
        try:
            q.put(array('h', stream.read(CHUNK_SIZE)))
        except Full:
            pass  # discard 

    
    return stream,sr
    


def record(stopped, q):
    while True:
        if stopped.wait(timeout=0):
            break
        chunk = q.get()
        vol = max(chunk)
        if vol >= MIN_VOLUME:
          pass
        else:
           ()
    

def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string


CHUNK_SIZE = 1024
MIN_VOLUME = 500
BUF_MAX_SIZE = CHUNK_SIZE * 10


def predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project):
    #get event and queue
    stopped = threading.Event()
    q = Queue(maxsize=int(round(BUF_MAX_SIZE / CHUNK_SIZE)))

    listen_t = threading.Thread(target=listen, args=(stopped, q))
    listen_t.start()
    record_t = threading.Thread(target=record, args=(stopped, q))
    record_t.start()

    try: 
            listen_t.join(0.1)
            record_t.join(0.1)
    except KeyboardInterrupt:
        stopped.set()

    listen_t.join()
    record_t.join()


    #collect new speech 
    speech,sr = listen
    #save sound
    recording_folder = "{}/recordings".format(head_folder_curr_project)
    if not os.path.exists(recording_folder):
        os.makedirs(recording_folder)
    
    timestamp = get_date()
    speech_filename = "{}/speech_{}.wav".format(recording_folder,timestamp)
    write(speech_filename,speech,sr)
    
    sr = librosa.load(speech_filename,sr=sr)
    
    
    
    features = featfun.coll_feats_manage_timestep(timesteps,frame_width,speech_filename,feature_type,num_filters,num_feature_columns,recording_folder)
    
    
    #need to reshape data for various models..
    #find out which models:
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
    return X


def main(project_head_folder,model_name):

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
    timesteps = int(feats_dict['timesteps'])
    context_window = int(feats_dict['context window'])
    frame_width = context_window*2+1
      
    #prepare the dictionary to find out the assigned label 
    with open(encoded_label_path, mode='r') as infile:
        reader = csv.reader(infile)
        label_for_dict = []  
        for rows in reader:
            label_for_dict.extend(rows)
        dict_labels_encoded = {label_for_dict[0]:label_for_dict[1] , label_for_dict[2]:label_for_dict[3], label_for_dict[4]:label_for_dict[5], label_for_dict[6]:label_for_dict[7], label_for_dict[8]:label_for_dict[9] }

    
    print("\nAvailable labels:")
    for key, value in dict_labels_encoded.items():
        print(value)
        
    
    #load model
    model = load_model(model_path)

    X = predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project)
    prediction = model.predict(X)
    pred = str(np.argmax(prediction[0]))    
    label = dict_labels_encoded[pred]
    print("Label without noise reduction: {}".format(label))

    if label == "ThoRaKhom":
        Y = predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project)
        prediction_2 = model.predict(Y)
        pred_2 = str(np.argmax(prediction_2[0]))
    
        label_2 = dict_labels_encoded[pred_2]
        print("Command is: {}".format(label_2))
    else:
        pass

    return None

if __name__=="__main__":
    
    project_head_folder = "mfcc13_models_2021y11m16d17h8m56s"
    model_name = "CNNLSTM_speech_commands001"
    while True:
        main(project_head_folder,model_name)
        
    
            
