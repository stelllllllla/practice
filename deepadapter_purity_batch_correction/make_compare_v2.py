import os
import numpy as np
import pandas as pd
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap

from deepadapter.utils import data_utils as DT
from deepadapter.models.dl_utils import AE

B = "DA_model/batch"
DIR2K  = os.path.join(B, "purity_official")
DIR30K = os.path.join(B, "purity_official_30000ep")
OUT = DIR30K

# ---- read v2 metrics for annotations ----
df = pd.read_csv(os.path.join(OUT, "fulldata_v2_res.csv"))
m = {r["condition"]: [float(r["align"]),float(r["asw_var"]),float(r["nmi"]),float(r["ari"])]
     for _,r in df.iterrows()}
m_base = m["Uncorrected_FULLv2"]; m_2k = m["Corrected2k_FULLv2"]; m_30k = m["Corrected30k_FULLv2"]
print("base",m_base,"2k",m_2k,"30k",m_30k, flush=True)

# ---- load + preprocess (official) ----
L = DT.LoadTransData()
dat1_raw, ids1 = L.load_data_purity("cl")
dat2_raw, ids2 = L.load_data_purity("tm")
P = DT.PrepTransData()
dat1_raw = P.sample_norm(dat1_raw); dat2_raw = P.sample_norm(dat2_raw)
data1, data2, sorted_cols = P.sort_genes(dat1_raw, dat2_raw)
data1 = P.sample_log(data1); data2 = P.sample_log(data2)
data = np.vstack((data1, data2))
unw = np.array([0]*len(data1)+[1]*len(data2))
dis_labels, diseases, d2l, l2d = L.load_bioLabel_purity(ids1, ids2)
unw2label, label2unw, labels, labels_onehot = P.label2onehot(unw)
n_cl = len(ids1)

# merge + filter (same as eval v2)
MERGE={"lymphocyte":"circulatory","blood":"circulatory","thymus":"circulatory"}
lin_m = np.array([MERGE.get(x,x) for x in diseases])
cl_m, tm_m = lin_m[:n_cl], lin_m[n_cl:]
kept=[l for l in sorted(set(lin_m)) if (cl_m==l).sum()>40 and (tm_m==l).sum()>40]
keep_mask = np.isin(lin_m, kept)
print("kept:", kept, "n_keep:", keep_mask.sum(), flush=True)

def corrected_all(ae_path):
    ae = AE(data.shape[1], 256, labels_onehot.shape[1], 128, 0.3).cuda()
    ae.load_state_dict(torch.load(ae_path)); ae.eval()
    outs=[]
    with torch.no_grad():
        for i in range(0,len(data),2000):
            x=torch.FloatTensor(data[i:i+2000]).cuda()
            outs.append(ae.dec(ae.enc(x)).cpu().numpy())
    return np.vstack(outs)

def run_umap(mat):
    return umap.UMAP(random_state=42, min_dist=0.99, metric="euclidean", n_neighbors=15).fit_transform(mat)

emb_unc = run_umap(data)
emb_2k  = run_umap(corrected_all(os.path.join(DIR2K,"ae.tar")))
emb_30k = run_umap(corrected_all(os.path.join(DIR30K,"ae.tar")))
print("umap done", flush=True)
np.savez(os.path.join(OUT,"compare_v2_embeddings.npz"),
         emb_unc=emb_unc, emb_2k=emb_2k, emb_30k=emb_30k,
         batch=unw, lin_m=lin_m, keep_mask=keep_mask,
         m_base=np.array(m_base), m_2k=np.array(m_2k), m_30k=np.array(m_30k))
print("saved npz", flush=True)

# ========== FIGURE 1: grouped bars (v2 metrics) ==========
conds=["Uncorrected","Corrected 2k ep","Corrected 30k ep"]
cc=["#999999","#4C9BD6","#D9622B"]
M=np.array([m_base,m_2k,m_30k])
fig,axes=plt.subplots(1,2,figsize=(13,5))
groups=[("Batch removal (higher = batch erased)",[0,1],["align","asw_var"]),
        ("Bio preservation (higher = lineage kept)",[2,3],["nmi","ari"])]
for ax,(title,cols,names) in zip(axes,groups):
    x=np.arange(len(names)); w=0.25
    for i,cond in enumerate(conds):
        ax.bar(x+(i-1)*w, M[i,cols], w, label=cond, color=cc[i], edgecolor="white")
        for j,c in enumerate(cols):
            ax.text(x[j]+(i-1)*w, M[i,c]+0.01, f"{M[i,c]:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11); ax.set_ylim(0,1.0)
    ax.set_title(title, fontsize=12); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Purity (author metric: merge circulatory, drop lineages with <=40 cl or tm)", fontsize=12)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(OUT,"fig1_bars_v2.png"), dpi=200, bbox_inches="tight")
print("saved fig1_bars_v2.png", flush=True)

# ========== FIGURE 2: 2x3 UMAP (batch row + lineage row) ==========
embs=[emb_unc,emb_2k,emb_30k]; cts=["Uncorrected","Corrected 2k ep","Corrected 30k ep"]; mm=[m_base,m_2k,m_30k]
bn={0:"cell line",1:"tumor"}; bc={0:"#2C7FB8",1:"#D95F02"}
cmap=plt.get_cmap("tab10"); lin_color={l:cmap(i) for i,l in enumerate(kept)}
fig,axes=plt.subplots(2,3,figsize=(19,12))
for c,(emb,ct,me) in enumerate(zip(embs,cts,mm)):
    axb=axes[0,c]
    for b in [0,1]:
        k=unw==b
        axb.scatter(emb[k,0],emb[k,1],s=6,alpha=0.55,c=bc[b],label=bn[b],linewidths=0)
    axb.set_title("%s  (by BATCH)\nalign=%.3f  asw_var=%.3f"%(ct,me[0],me[1]),fontsize=11)
    axb.legend(markerscale=2,fontsize=8); axb.set_xticks([]); axb.set_yticks([])
    axl=axes[1,c]
    nk=~keep_mask
    axl.scatter(emb[nk,0],emb[nk,1],s=5,alpha=0.25,c="#DDDDDD",label="dropped(<=40)",linewidths=0)
    for l in kept:
        k=keep_mask&(lin_m==l)
        axl.scatter(emb[k,0],emb[k,1],s=6,alpha=0.75,c=[lin_color[l]],label=l,linewidths=0)
    axl.set_title("%s  (by LINEAGE, 8 kept)\nnmi=%.3f  ari=%.3f"%(ct,me[2],me[3]),fontsize=11)
    axl.legend(markerscale=2,fontsize=7,ncol=2,loc="best"); axl.set_xticks([]); axl.set_yticks([])
fig.suptitle("Purity batch correction (author metric): Uncorrected vs 2k vs 30k  (n=%d, nmi/ari on %d kept lineages)"
             %(len(unw),len(kept)),fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(os.path.join(OUT,"fig2_umap_v2.png"), dpi=180, bbox_inches="tight")
print("saved fig2_umap_v2.png", flush=True)
print("DONE_FIGS_V2", flush=True)
