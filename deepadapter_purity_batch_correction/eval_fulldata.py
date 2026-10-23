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

# --------
L = DT.LoadTransData()
dat1_raw, ids1 = L.load_data_purity("cl")
dat2_raw, ids2 = L.load_data_purity("tm")
P = DT.PrepTransData()
dat1_raw = P.sample_norm(dat1_raw); dat2_raw = P.sample_norm(dat2_raw)
data1, data2, sorted_cols = P.sort_genes(dat1_raw, dat2_raw)
data1 = P.sample_log(data1); data2 = P.sample_log(data2)
data = np.vstack((data1, data2))
unw_variations = np.array([0]*len(data1) + [1]*len(data2))
dis_labels, diseases, d2l, l2d = L.load_bioLabel_purity(ids1, ids2)
unw2label, label2unw, labels, labels_onehot = P.label2onehot(unw_variations)
print("FULL data:", data.shape, "n_batch:", len(set(labels)), "n_lin:", len(set(diseases)), flush=True)

# ---- 新①:不划分,直接用全部 ----
all_batch = labels      # 批次整数标签,全部 13485
all_lin   = diseases    # 谱系字符串,全部 13485

# ---- 新②:对全部数据做 AE 推理 ----
def corrected_all(ae_path):
    ae = AE(data.shape[1], 256, labels_onehot.shape[1], 128, 0.3).cuda()
    ae.load_state_dict(torch.load(ae_path)); ae.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(data), 2000):
            x = torch.FloatTensor(data[i:i+2000]).cuda()
            outs.append(ae.dec(ae.enc(x)).cpu().numpy())
    return np.vstack(outs)

# ---- 新③:四指标(和 trainer.evaluate 同口径:UMAP→align/asw/nmi/ari) ----
def four_metrics(mat, tag):
    trans, align = DPU.decom_plot_nosplit(
        mat, all_batch, os.path.join(OUT, f"umap_FULL_{tag}.png"),
        colors=["#2C7FB8", "#D95F02"], label2name=label2unw, fitter="umap")
    nc = len(set(all_lin))
    nmi = UT.NMI_index(trans, all_lin, n_cluster=nc)
    ari = UT.ARI_index(trans, all_lin, n_cluster=nc)
    asw_var = 1 - UT.ASW_index(trans, all_batch)
    return align, asw_var, nmi, ari

# ---- 三条件各算一次 ----
rows = []
rows.append(("Uncorrected_FULL",)  + four_metrics(data, "uncorrected"))
rows.append(("Corrected2k_FULL",)  + four_metrics(corrected_all(os.path.join(DIR2K,  "ae.tar")), "2k"))
rows.append(("Corrected30k_FULL",) + four_metrics(corrected_all(os.path.join(DIR30K, "ae.tar")), "30k"))

rec = os.path.join(OUT, "fulldata_res.csv")
with open(rec, "w") as f:
    f.write("condition,align,asw_var,nmi,ari\n")
    for r in rows:
        f.write("%s,%s,%s,%s,%s\n" % r)
        print("%-20s align=%.4f asw_var=%.4f nmi=%.4f ari=%.4f" % r, flush=True)
print("DONE_FULLEVAL", flush=True)