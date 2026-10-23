import os
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")

from deepadapter.utils import data_utils as DT
import deepadapter.utils.utils as UT
import deepadapter.utils.decomposition_utils as DPU
from deepadapter.models.dl_utils import AE

B = "DA_model/batch"
DIR2K  = os.path.join(B, "purity_official")
DIR30K = os.path.join(B, "purity_official_30000ep")
OUT = DIR30K

# ---- load + preprocess (official, same as train_tumor.py) ----
L = DT.LoadTransData()
dat1_raw, ids1 = L.load_data_purity("cl")   # cl = dat1
dat2_raw, ids2 = L.load_data_purity("tm")   # tm = dat2
P = DT.PrepTransData()
dat1_raw = P.sample_norm(dat1_raw); dat2_raw = P.sample_norm(dat2_raw)
data1, data2, sorted_cols = P.sort_genes(dat1_raw, dat2_raw)
data1 = P.sample_log(data1); data2 = P.sample_log(data2)
data = np.vstack((data1, data2))
unw_variations = np.array([0]*len(data1) + [1]*len(data2))
dis_labels, diseases, d2l, l2d = L.load_bioLabel_purity(ids1, ids2)
unw2label, label2unw, labels, labels_onehot = P.label2onehot(unw_variations)
all_batch = labels
all_lin   = np.array(diseases)
n_cl = len(ids1)

# ===== author detail 1: merge lymphocyte/blood/thymus -> circulatory =====
MERGE = {"lymphocyte": "circulatory", "blood": "circulatory", "thymus": "circulatory"}
merge_lin = lambda a: np.array([MERGE.get(x, x) for x in a])
all_lin_m = merge_lin(all_lin)

# ===== author detail 2: drop a lineage if cl<=40 OR tm<=40 =====
cl_lin_m = all_lin_m[:n_cl]
tm_lin_m = all_lin_m[n_cl:]
kept = [l for l in sorted(set(all_lin_m))
        if (cl_lin_m == l).sum() > 40 and (tm_lin_m == l).sum() > 40]
keep_mask = np.isin(all_lin_m, kept)
print("kept lineages:", len(kept), kept, flush=True)
print("samples for nmi/ari:", keep_mask.sum(), "/", len(all_lin_m), flush=True)

def corrected_all(ae_path):
    ae = AE(data.shape[1], 256, labels_onehot.shape[1], 128, 0.3).cuda()
    ae.load_state_dict(torch.load(ae_path)); ae.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(data), 2000):
            x = torch.FloatTensor(data[i:i+2000]).cuda()
            outs.append(ae.dec(ae.enc(x)).cpu().numpy())
    return np.vstack(outs)

def four_metrics(mat, tag):
    trans, align = DPU.decom_plot_nosplit(
        mat, all_batch, os.path.join(OUT, f"umap_FULLv2_{tag}.png"),
        colors=["#2C7FB8", "#D95F02"], label2name=label2unw, fitter="umap")
    asw_var = 1 - UT.ASW_index(trans, all_batch)            # batch: ALL data
    sub_emb = trans[keep_mask]                              # bio: merged + filtered subset
    sub_lin = all_lin_m[keep_mask]
    nc = len(set(sub_lin))
    nmi = UT.NMI_index(sub_emb, sub_lin, n_cluster=nc)
    ari = UT.ARI_index(sub_emb, sub_lin, n_cluster=nc)
    return align, asw_var, nmi, ari

rows = []
rows.append(("Uncorrected_FULLv2",)  + four_metrics(data, "uncorrected"))
rows.append(("Corrected2k_FULLv2",)  + four_metrics(corrected_all(os.path.join(DIR2K,  "ae.tar")), "2k"))
rows.append(("Corrected30k_FULLv2",) + four_metrics(corrected_all(os.path.join(DIR30K, "ae.tar")), "30k"))

rec = os.path.join(OUT, "fulldata_v2_res.csv")
with open(rec, "w") as f:
    f.write("condition,align,asw_var,nmi,ari\n")
    for r in rows:
        f.write("%s,%s,%s,%s,%s\n" % r)
        print("%-22s align=%.4f asw_var=%.4f nmi=%.4f ari=%.4f" % r, flush=True)
print("DONE_FULLEVAL_V2", flush=True)
