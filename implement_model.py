import os
import csv
import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from keras.models import load_model
import time as tm
import sys
from queue import Queue
import feature_extraction_scripts.feature_extraction_functions as featfun
import feature_extraction_scripts.prep_noise as pn
import pyaudio
from time import time
def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day,time.hour,time.minute,time.second)
    return(time_str)

# feed_samples=64000
# q = Queue()
# sound = np.zeros(feed_samples, dtype='int16')
# run = True
# silence_threshold = 100
# duration = 5.5  # seconds
# times = list()
# duration = 5.5

# def callback(indata, outdata, frames, time, status):
#     global times
#     if status:
#         print("stop")
#         times.append(tm.time())
#     outdata[:] = indata
# with sd.RawStream(channels=1, dtype='int24', callback=callback):
#     sd.sleep(int(duration * 1000))
    
    
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def record_sound(sec,message):
    duration = 5.5
    sr = 16000
    print(message+" for {} seconds..".format(sec))
    sound = sd.rec(int(sec*sr),samplerate=sr,channels=1)
    sd.wait()
    return sound, sr

  
    

def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string

def predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project):
    #collect new speech 
    speech,sr = record_sound(4,"Please say *loud and clear* one of the target words. \nRecording")
    #save sound
    recording_folder = "{}/recordings".format(head_folder_curr_project)
    if not os.path.exists(recording_folder):
        os.makedirs(recording_folder)
    
    timestamp = get_date()
    speech_filename = "{}/speech_{}.wav".format(recording_folder,timestamp)
    sf.write(speech_filename,speech,sr)
    
    y_speech, sr = librosa.load(speech_filename,sr=sr)
    
    
    
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
        # print("x is=",X)
        # print("timestep=",timesteps)   
        # print("frame=",frame_width)    
        # print("x.shape=",X.shape[1])
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
   
    # star_record = time()
    X = predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project)
    prediction = model.predict(X)
    pred = str(np.argmax(prediction[0]))    
    label = dict_labels_encoded[pred]
 
    print("Label without noise reduction: {}".format(label))
    
    # if label == "ThoRaKhom":
    #     Y = predict(timesteps,frame_width,feature_type,num_filters,num_feature_columns,model_log_path,head_folder_curr_project)
    #     prediction_2 = model.predict(Y)
    #     pred_2 = str(np.argmax(prediction_2[0]))
    #     print("Label without noise reduction: {}".format(label))
    #     label_2 = dict_labels_encoded[pred_2]
    #     print("Command is: {}".format(label_2))      
        # probLabel = tf.nn.softmax(pred_2).np()
        # print(probLabel)
    # else:
    # print("It not a Wake up word please say it again")
    
    return None

if __name__=="__main__":
    
    project_head_folder = "mfcc40_models_2022y1m21d18h33m55s"
    model_name = "CNNLSTM_speech_commands.001"
    while True:
        main(project_head_folder,model_name)
        
    
            
