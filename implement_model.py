import os
import csv
import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
from keras.models import load_model

import feature_extraction_scripts.feature_extraction_functions as featfun

def get_date():
    time = datetime.datetime.now()
    time_str = "{}d{}h{}m{}s".format(time.day,time.hour,time.minute,time.second)
    return(time_str)

def record_sound(sec,message):
    sr = 16000
    print(message+" for {} seconds..".format(sec))
    sound = sd.rec(int(sec*sr),samplerate=sr,channels=1)
    sd.wait()
    return sound, sr

def str2bool(bool_string):
    bool_string = bool_string=="True"
    return bool_string

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
    
    features = featfun.coll_feats_manage_timestep(timesteps,frame_width,speech_filename,feature_type,num_filters,num_feature_columns,recording_folder,delta=delta,dom_freq=dom_freq)
    
    
    #reshape data for various models..
    #find out which models:
    with open(model_log_path, mode='r') as infile:
        reader = csv.reader(infile)            
        dict_model_settings = {rows[0]:rows[1] for rows in reader}
        
    model_type = dict_model_settings["model type"]
    
    

    X = features
    if model_type == "lstm":
        X = X.reshape((timesteps,frame_width,X.shape[1]))
    elif model_type == "cnn":
        X = X.reshape((X.shape[0],X.shape[1],1))
        X = X.reshape((1,)+X.shape)
    elif model_type == "cnnlstm":
        X = X.reshape((timesteps,frame_width,X.shape[1],1))
        X = X.reshape((1,)+X.shape)        
    
    
    #load model
    model = load_model(model_path)
    
    prediction = model.predict(X)
    pred = str(np.argmax(prediction[0]))
    
    label = dict_labels_encoded[pred]
    print("Label without noise reduction: {}".format(label))
    

    
    
    return None

if __name__=="__main__":
    
    project_head_folder = "mfcc13_models_2021y11m16d17h8m56s"
    model_name = "CNNLSTM_speech_commands001"
    
    main(project_head_folder,model_name)
        
    
            
