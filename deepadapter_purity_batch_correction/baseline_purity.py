import os
import pandas as pd
import numpy as np

from deepadapter.utils import data_utils as DT
import deepadapter.utils.utils as UT
import deepadapter.utils.decomposition_utils as DPU

# ---- load exactly like train_purity.py (cl first, tm second) ----
loadTransData = DT.LoadTransData()
cl_df, cl_ids = loadTransData.load_data_purity("cl")
tm_df, tm_ids = loadTransData.load_data_purity("tm")
data = pd.concat([cl_df, tm_df], axis=0, ignore_index=True)
ids = np.array(list(cl_ids) + list(tm_ids))
batches = np.array(["cell line"] * len(cl_ids) + ["tumor"] * len(tm_ids))
dis_labels, diseases, dis2label, label2dis = loadTransData.load_bioLabel_purity(cl_ids, tm_ids)

# ---- same preprocessing ----
prepTransData = DT.PrepTransData()
raw_df = prepTransData.sample_norm(data)
raw_df, sorted_cols = prepTransData.sort_genes_sgl_df(raw_df)
input_arr = prepTransData.sample_log(raw_df)
bat2label, label2bat, unwanted_labels, unwanted_onehot = prepTransData.label2onehot(batches)

# ---- same split (RandomState(0) inside -> identical test set) ----
(train_data, train_labels, train_labels_hot,
 val_data, val_labels, val_labels_hot,
 test_data, test_labels, test_labels_hot,
 train_ids, val_ids, test_ids,
 tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs) = DT.data_split_random(
    input_arr, unwanted_labels, unwanted_onehot, ids)

test_bios = diseases[tot_test_idxs]
test_unw = test_labels            # batch integer labels
test_wnt = test_bios              # disease strings
print("baseline test_num:", len(test_wnt), "n_bio:", len(set(test_wnt)), flush=True)

out_dir = "DA_model/batch/purity_tmcl/"
os.makedirs(out_dir, exist_ok=True)

# ---- UMAP on UNCORRECTED preprocessed test matrix (same fitter/seed as evaluate) ----
colors = ["#57904B", "violet", "#C93C2A", "#372A8F"]
rng = np.random.RandomState(0)
import matplotlib
tot_colors = list(matplotlib.colors.cnames.keys())
while len(colors) < max(len(set(test_unw)), len(set(test_wnt))):
    c = rng.choice(tot_colors, 1)[0]
    if c not in colors:
        colors.append(c)

label2bat_name = label2bat
bio_label2name = {t: t for t in set(diseases)}

trans_aligned, align = DPU.decom_plot_nosplit(
    test_data, test_unw,
    os.path.join(out_dir, "umap_BASELINE_variation.png"),
    colors=colors, title="baseline_variation", label2name=label2bat_name, fitter="umap")

n_cluster = len(set(test_wnt))
nmi = UT.NMI_index(trans_aligned, test_wnt, n_cluster=n_cluster)
ari = UT.ARI_index(trans_aligned, test_wnt, n_cluster=n_cluster)
asw_var = 1 - UT.ASW_index(trans_aligned, test_unw)

rec_path = os.path.join(out_dir, "baseline_res.csv")
with open(rec_path, "w") as f:
    f.write("test_name,test_num,align,asw_var,nmi,ari\n")
    f.write(f"Purity_UNCORRECTED,{len(test_wnt)},{align},{asw_var},{nmi},{ari}\n")
print(f"BASELINE  align={align:.4f}  asw_var={asw_var:.4f}  nmi={nmi:.4f}  ari={ari:.4f}", flush=True)
print("DONE_BASELINE", flush=True)
