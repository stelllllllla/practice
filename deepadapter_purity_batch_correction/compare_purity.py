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

OUT = "DA_model/batch/purity_tmcl/"

# ---- load + split exactly like train/baseline (RandomState(0) -> same test set) ----
L = DT.LoadTransData()
cl_df, cl_ids = L.load_data_purity("cl")
tm_df, tm_ids = L.load_data_purity("tm")
data = pd.concat([cl_df, tm_df], axis=0, ignore_index=True)
ids = np.array(list(cl_ids) + list(tm_ids))
batches = np.array(["cell line"] * len(cl_ids) + ["tumor"] * len(tm_ids))
dis_labels, diseases, dis2label, label2dis = L.load_bioLabel_purity(cl_ids, tm_ids)

P = DT.PrepTransData()
raw_df = P.sample_norm(data)
raw_df, sorted_cols = P.sort_genes_sgl_df(raw_df)
input_arr = P.sample_log(raw_df)
bat2label, label2bat, unw_labels, unw_onehot = P.label2onehot(batches)

(tr_d, tr_l, tr_lh, va_d, va_l, va_lh, te_d, te_l, te_lh,
 tr_ids, va_ids, te_ids,
 tv_idx, tr_idx, va_idx, te_idx) = DT.data_split_random(input_arr, unw_labels, unw_onehot, ids)
te_bios = diseases[te_idx]
print("test_num:", len(te_bios), "n_bio:", len(set(te_bios)), flush=True)

# ---- load trained AE, produce corrected matrix = dec(enc(x)) ----
net_hidden, net_z, net_drop = 256, 128, 0.3
in_dim = te_d.shape[1]; batch_dim = unw_onehot.shape[1]
ae = AE(in_dim, net_hidden, batch_dim, net_z, net_drop).cuda()
ae.load_state_dict(torch.load(os.path.join(OUT, "ae.tar")))
ae.eval()
with torch.no_grad():
    x = torch.FloatTensor(te_d).cuda()
    corrected = ae.dec(ae.enc(x)).detach().cpu().numpy()
print("corrected:", corrected.shape, flush=True)

# ---- UMAP (same params as library decom_plot_nosplit) ----
def run_umap(mat):
    reducer = umap.UMAP(random_state=42, min_dist=0.99, metric="euclidean", n_neighbors=15)
    return reducer.fit_transform(mat)

emb_unc = run_umap(te_d)
emb_cor = run_umap(corrected)
print("umap done", flush=True)

# ---- metrics for annotation (same funcs as evaluate) ----
def metrics(emb):
    from deepadapter.utils.utils import alignment_score
    al = alignment_score(emb, te_l)
    nc = len(set(te_bios))
    nmi = UT.NMI_index(emb, te_bios, n_cluster=nc)
    ari = UT.ARI_index(emb, te_bios, n_cluster=nc)
    asw = 1 - UT.ASW_index(emb, te_l)
    return al, asw, nmi, ari

m_unc = metrics(emb_unc)
m_cor = metrics(emb_cor)
print("BASELINE  align=%.3f asw=%.3f nmi=%.3f ari=%.3f" % m_unc, flush=True)
print("CORRECTED align=%.3f asw=%.3f nmi=%.3f ari=%.3f" % m_cor, flush=True)

# ---- save embeddings for local re-plotting if needed ----
np.savez(os.path.join(OUT, "compare_embeddings.npz"),
         emb_unc=emb_unc, emb_cor=emb_cor,
         batch_labels=te_l, bios=te_bios,
         m_unc=np.array(m_unc), m_cor=np.array(m_cor))

# ---- 2x2 figure: rows=color-by(batch/lineage), cols=(uncorrected/corrected) ----
batch_names = {0: "cell line", 1: "tumor"}
batch_colors = {0: "#2C7FB8", 1: "#D95F02"}

# top lineages by frequency; rest -> "other"
uniq, cnt = np.unique(te_bios, return_counts=True)
order = uniq[np.argsort(-cnt)]
TOPN = 10
top = list(order[:TOPN])
cmap = plt.get_cmap("tab10")
lin_color = {l: cmap(i) for i, l in enumerate(top)}
def lin_c(l): return lin_color[l] if l in lin_color else "#CCCCCC"

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

def scatter_batch(ax, emb, m, title):
    for b in [0, 1]:
        mask = te_l == b
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                   c=batch_colors[b], label=batch_names[b], linewidths=0)
    ax.set_title("%s\nalign=%.3f  asw_var=%.3f" % (title, m[0], m[1]), fontsize=12)
    ax.legend(markerscale=2, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

def scatter_lineage(ax, emb, m, title):
    other = ~np.isin(te_bios, top)
    if other.sum():
        ax.scatter(emb[other, 0], emb[other, 1], s=8, alpha=0.4, c="#CCCCCC",
                   label="other", linewidths=0)
    for l in top:
        mask = te_bios == l
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.7,
                   c=[lin_color[l]], label=l, linewidths=0)
    ax.set_title("%s\nnmi=%.3f  ari=%.3f" % (title, m[2], m[3]), fontsize=12)
    ax.legend(markerscale=2, fontsize=7, ncol=2, loc="best"); ax.set_xticks([]); ax.set_yticks([])

scatter_batch(axes[0, 0], emb_unc, m_unc, "Uncorrected  (by BATCH)")
scatter_batch(axes[0, 1], emb_cor, m_cor, "DeepAdapter corrected  (by BATCH)")
scatter_lineage(axes[1, 0], emb_unc, m_unc, "Uncorrected  (by LINEAGE)")
scatter_lineage(axes[1, 1], emb_cor, m_cor, "DeepAdapter corrected  (by LINEAGE)")

fig.suptitle("Purity (tumor vs cell line) batch correction  -  test n=%d, %d lineages"
             % (len(te_bios), len(set(te_bios))), fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(os.path.join(OUT, "compare_purity.png"), dpi=200, bbox_inches="tight")
print("saved compare_purity.png", flush=True)
print("DONE_COMPARE", flush=True)
