import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from deepadapter.utils import data_utils as DT
import deepadapter.utils.utils as UT
from deepadapter.models.dl_utils import AE

B = "DA_model/batch"
DIR2K = os.path.join(B, "purity_official")
DIR30K = os.path.join(B, "purity_official_30000ep")
OUT = DIR30K   # save figures here

# ---- read official evaluate metrics (the numbers we quote) ----
def read_metrics(path):
    df = pd.read_csv(path)
    r = df.iloc[0]
    return [float(r["align"]), float(r["asw_var"]), float(r["nmi"]), float(r["ari"])]
m_base = read_metrics(os.path.join(DIR2K, "baseline_official_res.csv"))
m_2k   = read_metrics(os.path.join(DIR2K, "test_res.csv"))
m_30k  = read_metrics(os.path.join(DIR30K, "test_res.csv"))
print("base:", m_base, "\n2k  :", m_2k, "\n30k :", m_30k, flush=True)

# ---- load data EXACTLY like train_tumor.py (official preproc) ----
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
ids = np.array(ids1 + ids2)

(tr_d,tr_l,tr_lh, va_d,va_l,va_lh, te_d,te_l,te_lh,
 tr_ids,va_ids,te_ids, tv,tri,vai,tei) = DT.data_split_random(
    data, labels, labels_onehot, ids, val_ratio=0.125)
te_batch = te_l                      # batch int labels (0=cl,1=tm)
te_lin   = diseases[tei]             # lineage strings
print("test:", te_d.shape, "n_lin:", len(set(te_lin)), flush=True)

# ---- AE inference: corrected = dec(enc(x)) ----
def corrected(ae_path):
    ae = AE(te_d.shape[1], 256, labels_onehot.shape[1], 128, 0.3).cuda()
    ae.load_state_dict(torch.load(ae_path)); ae.eval()
    with torch.no_grad():
        x = torch.FloatTensor(te_d).cuda()
        out = ae.dec(ae.enc(x)).detach().cpu().numpy()
    return out
cor2k  = corrected(os.path.join(DIR2K, "ae.tar"))
cor30k = corrected(os.path.join(DIR30K, "ae.tar"))
print("inference done", flush=True)

# ---- UMAP (same params as library) ----
def run_umap(mat):
    return umap.UMAP(random_state=42, min_dist=0.99, metric="euclidean", n_neighbors=15).fit_transform(mat)
emb_unc = run_umap(te_d)
emb_2k  = run_umap(cor2k)
emb_30k = run_umap(cor30k)
print("umap done", flush=True)

np.savez(os.path.join(OUT, "compare3_embeddings.npz"),
         emb_unc=emb_unc, emb_2k=emb_2k, emb_30k=emb_30k,
         te_batch=te_batch, te_lin=te_lin,
         m_base=np.array(m_base), m_2k=np.array(m_2k), m_30k=np.array(m_30k))
print("saved npz", flush=True)

# ========== FIGURE 1: grouped bar chart (4 metrics x 3 conditions) ==========
conds = ["Uncorrected", "Corrected 2k ep", "Corrected 30k ep"]
cond_colors = ["#999999", "#4C9BD6", "#D9622B"]
M = np.array([m_base, m_2k, m_30k])   # rows=cond, cols=[align,asw_var,nmi,ari]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# left: batch removal (align, asw_var); right: bio preservation (nmi, ari)
groups = [("Batch removal (higher = batch erased)", [0,1], ["align","asw_var"]),
          ("Bio preservation (higher = lineage kept)", [2,3], ["nmi","ari"])]
for ax,(title,cols,names) in zip(axes, groups):
    x = np.arange(len(names)); w = 0.25
    for i,cond in enumerate(conds):
        ax.bar(x + (i-1)*w, M[i, cols], w, label=cond, color=cond_colors[i], edgecolor="white")
        for j,c in enumerate(cols):
            ax.text(x[j]+(i-1)*w, M[i,c]+0.01, f"{M[i,c]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.0); ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Purity (tumor vs cell line): metric comparison across training length", fontsize=13)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT, "fig1_bars_compare3.png"), dpi=200, bbox_inches="tight")
print("saved fig1_bars_compare3.png", flush=True)

# ========== FIGURE 2: 2x3 UMAP grid ==========
embs = [emb_unc, emb_2k, emb_30k]
col_titles = ["Uncorrected", "Corrected 2k ep", "Corrected 30k ep"]
mets = [m_base, m_2k, m_30k]
batch_names = {0:"cell line", 1:"tumor"}
batch_colors = {0:"#2C7FB8", 1:"#D95F02"}
uniq,cnt = np.unique(te_lin, return_counts=True)
top = list(uniq[np.argsort(-cnt)][:10])
cmap = plt.get_cmap("tab10")
lin_color = {l:cmap(i) for i,l in enumerate(top)}

fig, axes = plt.subplots(2, 3, figsize=(19, 12))
for c,(emb,ct,mm) in enumerate(zip(embs, col_titles, mets)):
    axb = axes[0,c]
    for b in [0,1]:
        mk = te_batch==b
        axb.scatter(emb[mk,0], emb[mk,1], s=7, alpha=0.6, c=batch_colors[b], label=batch_names[b], linewidths=0)
    axb.set_title("%s  (by BATCH)\nalign=%.3f  asw_var=%.3f" % (ct, mm[0], mm[1]), fontsize=11)
    axb.legend(markerscale=2, fontsize=8); axb.set_xticks([]); axb.set_yticks([])
    axl = axes[1,c]
    other = ~np.isin(te_lin, top)
    if other.sum():
        axl.scatter(emb[other,0], emb[other,1], s=7, alpha=0.35, c="#CCCCCC", label="other", linewidths=0)
    for l in top:
        mk = te_lin==l
        axl.scatter(emb[mk,0], emb[mk,1], s=7, alpha=0.7, c=[lin_color[l]], label=l, linewidths=0)
    axl.set_title("%s  (by LINEAGE)\nnmi=%.3f  ari=%.3f" % (ct, mm[2], mm[3]), fontsize=11)
    axl.legend(markerscale=2, fontsize=6, ncol=2, loc="best"); axl.set_xticks([]); axl.set_yticks([])
fig.suptitle("Purity batch correction: Uncorrected vs 2k ep vs 30k ep  (test n=%d, %d lineages)"
             % (len(te_lin), len(set(te_lin))), fontsize=14)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(os.path.join(OUT, "fig2_umap_compare3.png"), dpi=180, bbox_inches="tight")
print("saved fig2_umap_compare3.png", flush=True)
print("DONE_FIGS", flush=True)
