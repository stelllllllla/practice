import os
import pandas as pd
import numpy as np

from deepadapter.utils import data_utils as DT
import deepadapter.utils.utils as UT
import deepadapter.utils.decomposition_utils as DPU

# ===== import EXACTLY like train_tumor.py (official reference) =====
### load data
dat1_name = "cl"; dat2_name = "tm"
loadTransData = DT.LoadTransData()
dat1_raw, ids1 = loadTransData.load_data_purity(dat1_name)
dat2_raw, ids2 = loadTransData.load_data_purity(dat2_name)

### preprocessing (SEPARATE norm -> sort_genes cl-pivot -> SEPARATE log)
prepTransData = DT.PrepTransData()
dat1_raw = prepTransData.sample_norm(dat1_raw)
dat2_raw = prepTransData.sample_norm(dat2_raw)
data1, data2, sorted_cols = prepTransData.sort_genes(dat1_raw, dat2_raw)
data1 = prepTransData.sample_log(data1)
data2 = prepTransData.sample_log(data2)

### assemble
data = np.vstack((data1, data2))
unw_variations = np.array([0]*len(data1) + [1]*len(data2))
dis_labels, diseases, disease2label, label2disease = loadTransData.load_bioLabel_purity(ids1, ids2)
unw2label, label2unw, labels, labels_onehot = prepTransData.label2onehot(unw_variations)
ids = np.array(ids1 + ids2)

### split (val_ratio=0.125, same as train_tumor.py)
(train_data, train_labels, train_labels_hot,
 val_data, val_labels, val_labels_hot,
 test_data, test_labels, test_labels_hot,
 train_ids, val_ids, test_ids,
 tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs) = DT.data_split_random(
    data, labels, labels_onehot, ids, val_ratio=0.125)
test_bios = dis_labels[tot_test_idxs]      # INT codes, like train_tumor.py

out_dir = "DA_model/batch/purity_official/"
os.makedirs(out_dir, exist_ok=True)
print("baseline_official test_num:", len(test_bios), "n_bio:", len(set(test_bios)), flush=True)

# ===== metrics on UNCORRECTED test matrix (same funcs/seed as trainer.evaluate) =====
test_unw = test_labels     # batch int labels
test_wnt = test_bios       # disease int labels

import matplotlib
colors = ["#57904B", "violet", "#C93C2A", "#372A8F"]
rng = np.random.RandomState(0)
tot_colors = list(matplotlib.colors.cnames.keys())
while len(colors) < max(len(set(test_unw)), len(set(test_wnt))):
    c = rng.choice(tot_colors, 1)[0]
    if c not in colors: colors.append(c)

trans_aligned, align = DPU.decom_plot_nosplit(
    test_data, test_unw,
    os.path.join(out_dir, "umap_BASELINE_official_variation.png"),
    colors=colors, title="baseline_official_variation", label2name=label2unw, fitter="umap")

n_cluster = len(set(test_wnt))
nmi = UT.NMI_index(trans_aligned, test_wnt, n_cluster=n_cluster)
ari = UT.ARI_index(trans_aligned, test_wnt, n_cluster=n_cluster)
asw_var = 1 - UT.ASW_index(trans_aligned, test_unw)

rec = os.path.join(out_dir, "baseline_official_res.csv")
with open(rec, "w") as f:
    f.write("test_name,test_num,align,asw_var,nmi,ari\n")
    f.write(f"Purity_UNCORR_official,{len(test_wnt)},{align},{asw_var},{nmi},{ari}\n")
print(f"BASELINE_OFFICIAL align={align:.4f} asw_var={asw_var:.4f} nmi={nmi:.4f} ari={ari:.4f}", flush=True)
print("DONE_BASELINE_OFFICIAL", flush=True)
