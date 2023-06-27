from tools.MultiBranchCNN import *
from tensorflow import keras

def predictSequenceFromSaveKerasMdl(test_path, mdl_path, testRs_path, ft_list=def_fts):
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
    fastas.to_csv(testRs_path, index=False)
    print("save %s prediction to: %s" % (test_path, testRs_path))
    return pred

if __name__ == "__main__":
    # generate test features
    test_path = "/home/yanjielu/MBC-Attention/test.fasta"
    mdl_path = "/home/yanjielu/MBC-Attention/model/whole_train.mdl"
    predict_test_path = "/home/yanjielu/MBC-Attention/test_pred.csv"
    pred = predictSequenceFromSaveKerasMdl(test_path, mdl_path, predict_test_path)
