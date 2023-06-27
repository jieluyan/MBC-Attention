from tools_GeneFt import *
from tools_MLmdl import *
import os
import pandas as pd
# s = ["escherichia coli", "staphylococcus aureus"]
# specie = s[0]
# all_ft_names = geneAllFeatureNames(low=1, high=19)
# # all_ft_names = ['AAC','type7raac2']
# ft_names = {"staphylococcus aureus": "AAC"}
# for ft_name in all_ft_names:
#     ft_names[specie] = ft_name
#     d2 = develop2SingleMdls(ft_whole_name=ft_names)
#     dlp_mdls = d2.developMdl(specie_name=specie)

all_ft_names = geneAllFeatureNames(low=1, high=19)
#all_ft_names = ['AAC', "DPC", "DDE", 'type1raac2glmd2g-gap']
from_start = False
log_path = os.path.join(GetFolder().log_dir, "train.log")
log_existed = os.path.isfile(log_path)
colnames = ["ft_name", "mdl_path", "train_info_path"]

if (not log_existed) or from_start:
    ft_hdl_infos = pd.DataFrame([],columns=colnames)
else:
    ft_hdl_infos = pd.read_csv(log_path)
    ft_names = ft_hdl_infos["ft_name"]
    for ft in ft_names:
        all_ft_names.remove(ft)
for ft_name in all_ft_names:
    print("Processing: ", ft_name)
    d1 = develop1MergeMdl(ft_whole_name=ft_name, target="EC_pMIC")
    dlp_mdls = d1.developMdl()
    l = [[ft_name, d1.mdl_path, d1.train_info_path]]
    t_df = pd.DataFrame(l, columns=colnames)
    ft_hdl_infos = pd.concat((ft_hdl_infos, t_df))
    ft_hdl_infos.to_csv(log_path, index=False)