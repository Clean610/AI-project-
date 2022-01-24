from train_models_CNN_LSTM_CNNLSTM import main
import tensorflow as tf
tf.random.set_seed(42)
# import the necessary packages
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from model_scripts.generator_speech_CNN_LSTM import Generator
import csv
# wrap our model into a scikit-learn compatible classifier
print("[INFO] initializing model...")
model = KerasClassifier(build_fn=main, verbose=0)
# define a grid of the hyperparameter search space
hiddenLayerOne = [256, 512, 784]
hiddenLayerTwo = [128, 256, 512]
learnRate = [1e-2, 1e-3, 1e-4]
dropout = [0.3, 0.4, 0.5]
batchSize = [4, 8, 16, 32]
epochs = [10, 20, 30, 40]
# create a dictionary from the hyperparameter grid
grid = dict(
	hiddenLayerOne=hiddenLayerOne,
	learnRate=learnRate,
	hiddenLayerTwo=hiddenLayerTwo,
	dropout=dropout,
	batch_size=batchSize,
	epochs=epochs
)



load_feature_settings_file = "ml_speech_projects/mfcc13_models_2021y11m16d17h8m56s/features_log.csv"
with open(load_feature_settings_file, mode='r') as infile:
        reader = csv.reader(infile)            
        feats_dict = {rows[0]:rows[1] for rows in reader}
    
num_labels = int(feats_dict['num classes'])

filename_train = "ml_speech_projects/mfcc13_models_2021y11m16d17h8m56s/data_train/train_features.npy" 
train_data = np.load(filename_train)  

filename_val = "ml_speech_projects/mfcc13_models_2021y11m16d17h8m56s/data_val/val_features.npy"
val_data = np.load(filename_val)

train_generator = Generator(model_type,train_data,timesteps,frame_width)
val_generator = Generator(model_type,val_data,timesteps,frame_width)




print("[INFO] performing random search...")
searcher = RandomizedSearchCV(estimator=model, n_jobs=-1, cv=3,
	param_distributions=grid, scoring="accuracy")
searchResults = searcher.fit(train_data, num_labels)
# summarize grid search information
bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,
	bestParams))

print("[INFO] evaluating the best model...")
bestModel = searchResults.best_estimator_
accuracy = bestModel.score(val_data, num_labels)
print("accuracy: {:.2f}%".format(accuracy * 100))