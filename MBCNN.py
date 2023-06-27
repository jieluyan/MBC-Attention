from tools.tools_MBCNNmdl import *

# if MIC  threshold=10 then test_ratio 20, val_ratio 10,
# if thresthold is 10000 then test_ratio 20, val_ratio 20,
# names = getMergeFastaColnames(host_specie=host_specie, target_specie=target_specie)
# target = names.MIC_name1
best30_pmic4 = \
['type8raac9glmd3lambda-correlation', 'type8raac7glmd3lambda-correlation', 'QSOrder_lmd4', 'QSOrder_lmd3', 'QSOrder_lmd2',
 'QSOrder_lmd1', 'QSOrder_lmd0', 'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
 'type5raac8glmd2lambda-correlation', 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation',
 'type2raac8glmd2lambda-correlation', 'type8raac14glmd1lambda-correlation', 'type8raac14glmd0g-gap', 'type7raac10glmd2lambda-correlation',
 'type14raac10glmd3lambda-correlation', 'type2raac15glmd3lambda-correlation', 'type7raac13glmd2lambda-correlation',
 'type3Braac12glmd3lambda-correlation', 'type5raac15glmd3lambda-correlation', 'type3Braac12glmd0g-gap', 'type3Braac12glmd1lambda-correlation',
 'type16raac15glmd4lambda-correlation', 'type3Braac12glmd4lambda-correlation', 'type7raac12glmd5lambda-correlation', 'type8raac14glmd3lambda-correlation',
 'type8raac15glmd4lambda-correlation', 'type2raac15glmd2lambda-correlation', 'type5raac15glmd2lambda-correlation']

best30_pmic4_uni = \
['type8raac9glmd3lambda-correlation', 'QSOrder_lmd4',  'type5raac15glmd4lambda-correlation', 'type7raac10glmd3lambda-correlation',
 'type3Braac9glmd3lambda-correlation', 'type2raac15glmd4lambda-correlation', 'type14raac10glmd3lambda-correlation',
 'type16raac15glmd4lambda-correlation']

def_lrParas = {"lr": 1e-4 ,"decay_rate": 0.9 ,"decay_steps": 100, "patience": 20, "monitor": "loss"}
cal_lrParas = {"lr": 1e-4 ,"decay_rate": 0.86 ,"decay_steps": 50, "patience": 20, "monitor": "loss"}
lrParasDict = {"sodium": def_lrParas, "potassium": def_lrParas, "calcium": cal_lrParas}
hyperParaDict = {"layer_num": 1, "dropOutRate": 0.4, "filter_num": 32, "whole_tune_epoch": 20, "epoch": 100}

def tuneMultiFts(bestFts_list, start=1, end=30):
    all_aves = []
    log_name = "EC_pMIC4_scale6_bias3_8_rep3-layer1-dropout0.4-filter32.rs"
    # server = 3
    # s, e = 5 * (server-1) + 1, 5 * server + 1
    s, e = start, end
    fld = GetFolder()
    log_dir = fld.log_dir
    name = "best_%d-%d_MBCNN-%s" % (s, e, log_name)
    path = os.path.join(log_dir, name)
    for i in range(s, e, 1):
        ft_list = bestFts_list[:i]
        # ave_mts, all_mts = MBCNN3timesValidation(ft_name=ft_list, host_specie="escherichia coli", target_specie="staphylococcus aureus",
        #         data_method="geneMergeFts", do_test=True, plot_scatter=True, font=10, onlyMrg=False, cdhit=False, test_ratio=20, val_ratio=10,
        #         process="host", repeat=3, test_rep=True, target="EC_pMIC", lrParas=def_lrParas, hyperParas = hyperParaDict,
        #         mdl_name=None, hist_name=None, log_name=log_name, enable_earlyStop=True)
        ave_mts, all_mts = MBCNN3timesValidationEC(ft_name=ft_list, specie="escherichia coli", do_test=True, plot_scatter=True,
                            font=9, edge=(-4, 4), repeat=3, target="EC_pMIC", lrParas=def_lrParas,
                            hyperParas = hyperParaDict, enable_earlyStop=False, threshold=largest_MIC, bias=def_bias,
                            scale=def_scale, mdl_name=None, hist_name=None, log_name=log_name)
        all_aves.append(ave_mts)
        rs = pd.concat(all_aves)
        print("MBCNN best_%d to best_%d average metrics: \n" % (s, i), rs)
        rs.to_csv(path, index=False)
        print("all average metrics saved to: \n", path)
    return rs
# best30_pmic4_uni or best30_pmic4
ave_mts = tuneMultiFts(bestFts_list=best30_pmic4, start=1, end=30)

"""
log_name = "GridSearchLossMSLE_layers_Drops_filter_nums_EC_pMIC4_scale6_bias3_8_rep3stop200epochs.rs"
all_aves = []
fld = GetFolder()
log_dir = fld.log_dir
for layer_num in range(1,4,1):
    hyperParaDict["layer_num"] = layer_num
    for rate in range(6):
        hyperParaDict["dropOutRate"] = rate * 0.1
        for filter_num in range(8):
            hyperParaDict["filter_num"] = (filter_num+1)*16
            paras = ["%s_%s" % (i, str(hyperParaDict[i])) for i in hyperParaDict]
            name = "GridSearch_" + "-".join(paras)
            ave_mts, all_mts = MBCNN3timesValidation(ft_name=best30_pmic4[:2], host_specie="escherichia coli",
                                             target_specie="staphylococcus aureus",
                                             data_method="geneMergeFts", do_test=True, plot_scatter=True, font=10,
                                             onlyMrg=False, cdhit=False, test_ratio=20, val_ratio=10,
                                             process="host", repeat=3, test_rep=True, target="EC_pMIC",
                                             lrParas=def_lrParas, hyperParas=hyperParaDict, enable_earlyStop=True,
                                             mdl_name="%s.mdl" % name, hist_name="%s.hist" % name, log_name=log_name)
            all_aves.append(ave_mts)
            rs = pd.concat(all_aves)
            print("************Tunning with out early stopping \n\tlayer_num: ", layer_num, "\n\tdrop out rate: ", rate*0.1,
                  "\n\t filter_num", (filter_num+1)*16, "\n\ttotal epochs: ", hyperParaDict["epoch"])
            print("************processing \n\tlearning rate: ", def_lrParas["lr"], "\n\tdecay rate: ", def_lrParas["decay_rate"],
                  "\n\tdecay step: ", def_lrParas["decay_steps"])
            path = os.path.join(log_dir, log_name)
            rs.to_csv(path, index=False)
            print("processed average metrics saved to: \n", path)
"""
