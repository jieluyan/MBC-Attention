# MBC_Attention
After download the project, please follow the requirements to deploy the environment.
If you want to test you sequences, put your sequences to a fasta file.
Then do the following codes in the python console to test your own sequences:
    from test_mbc_attention import *
    mdl_path = "yourPath/MBC-Attention/model/whole_train.mdl" # trained model 
    test_path = "yourFolderName/test.fasta" # input sequences path
    predict_test_path = "yourFolderName/test_pred.csv" # output path of the prediction
    pred = predictSequenceFromSaveKerasMdl(test_path, mdl_path, predict_test_path)
