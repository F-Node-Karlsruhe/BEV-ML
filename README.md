# BEV-ML

Implementation of seminar Thesis IIP

## Requirements
+ tensorflow 2.0.0^
+ pandas
+ matplotlib
+ numpy

## Installation
Follow these commands after cloning the repo.

### Install dependencies
    pip install tensorflow pandas matplotlib

In case tensorflow cannot be installed with pip go to <https://www.tensorflow.org/install/>

### Prepare data
The raw data must be found under ```data/raw.csv``` in the project folder.  
Run ```data_prep.py```  

    python data_prep.py

### Run model
Check the desired parameters in ```LSTM.py```.  
If you want to predict set ```TRAIN=False``` and make sure that the model is already available in ```models/```.  
If the model is not yet trained, set ```TRAIN=True``` and run the script.  

    python LSTM.py