# bd4h_final_project_GAMENet

Dependencies
python 3.7
torch 1.11

Instructions:
- run process_data.py (make sure the mimic3 csv files PRESCRIPTIONS, DIAGNOSES_ICD, PROCEDURES_ICD are in the data folder, they are not included in the github repository)
- this should generate the required input files for training the model
- gamenet.py contains the model itself
- run train_GAMEnet.py, this will train and output results