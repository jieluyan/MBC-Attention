from tools.base import *
from tools.features import *
from tools.MultiBranchCNN import *
import tensorflow as tf
from tensorflow import keras
from keras import optimizers
from keras.optimizers import schedules
from keras import layers, Sequential, losses, metrics, models
from keras.models import Model
from keras import backend as K
# TF_ENABLE_ONEDNN_OPTS=0

lrParas = {"lr": 5e-4 ,"decay_rate": 0.92, "decay_steps": 25, "patience": 15, "monitor": "loss"}
hyperParaDict = {"layer_num": 1, "dropOutRate": 0.4, "filter_num": 32, "epoch": 200}
best14 = \
['type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2',
 'QSOrder_lmd1', 'QSOrder_lmd0', 'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
 'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation',
 'type2raac8glmd2lambda-correlation', 'type8raac14glmd1lambda-correlation']

def predictSequenceFromSaveKerasMdl(test_path, mdl_path, testRs_path, ft_list=best14):
    test_fastas, test_fastas_file = geneFastasFromFastaFile(test_path)
    fastas = pd.read_csv(test_fastas_file)
    sets = CNNimportFtsDataSetsPoNe(fastas, ft_list=ft_list, target=None)
    names, seqs, lens, records = readFastaYan(test_path)
    X, Y = CNNstandardInputOutput(sets)
    mdl_name = os.path.basename(mdl_path)
    CNNMdl = keras.models.load_model(mdl_path)
    pred = CNNMdl.predict(X)
    pred = pred / def_scale - def_bias
    fastas["Prediction"] = 10 ** (-pred)
    fastas.to_csv(testRs_path, header=False)
    print("save %s prediction to: %s" % (test_path, testRs_path))
    return pred

if __name__ == "__main__":
    # generate test features
    test_path = "/home/yanjielu/MBC-Attention/test.fasta"
    mdl_path = "/home/yanjielu/MBC-Attention/model/whole_train.mdl"
    predict_test_path = "/home/yanjielu/MBC-Attention/test_pred.csv"
    pred = predictSequenceFromSaveKerasMdl(test_path, mdl_path, predict_test_path)
