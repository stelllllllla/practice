import os, sys
import pandas as pd
import numpy as np
import torch

from deepadapter.utils import data_utils as DT
from deepadapter.params import dl_params_purity as DLPARAM
from deepadapter.run import run as RUN
from deepadapter.models.data_loader import TransData, DataLoader

print("CUDA avail:", torch.cuda.is_available(),
      "device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      flush=True)

# ---- load purity: cell line (cl) + tumor (tm) ----
loadTransData = DT.LoadTransData()
cl_df, cl_ids = loadTransData.load_data_purity("cl")
tm_df, tm_ids = loadTransData.load_data_purity("tm")
print("cl:", cl_df.shape, "tm:", tm_df.shape, flush=True)

# combine: cell line FIRST, then tumor (must match load_bioLabel_purity id order)
data = pd.concat([cl_df, tm_df], axis=0, ignore_index=True)
ids = np.array(list(cl_ids) + list(tm_ids))
batches = np.array(["cell line"] * len(cl_ids) + ["tumor"] * len(tm_ids))
print("combined:", data.shape, "n_batches:", len(set(batches)), flush=True)

# biological signal = cancer lineage (cl: CL_tumor_class, tm: lineage)
dis_labels, diseases, dis2label, label2dis = loadTransData.load_bioLabel_purity(cl_ids, tm_ids)
print("n_bio(lineage):", len(set(diseases)), flush=True)

# ---- preprocess (on the COMBINED df so gene sort order is shared) ----
prepTransData = DT.PrepTransData()
raw_df = prepTransData.sample_norm(data)
raw_df, sorted_cols = prepTransData.sort_genes_sgl_df(raw_df)
input_arr = prepTransData.sample_log(raw_df)
bat2label, label2bat, unwanted_labels, unwanted_onehot = prepTransData.label2onehot(batches)

# ---- params (lambda_batch injected via env for the ablation sweep) ----
net_args = DLPARAM.load_dl_params()
net_args.epochs = int(os.environ.get("DA_EPOCHS", "2000"))
lb_str = os.environ.get("DA_LAMBDA_BATCH", "0.01")
net_args.lambda_batch = float(lb_str)
print("epochs:", net_args.epochs, "batch_size:", net_args.batch_size,
      "lambda_batch:", net_args.lambda_batch, "lambda_ae:", net_args.lambda_ae, flush=True)

# ---- split (RANDOM; official platform-split gives EMPTY test since cl n tm = 0) ----
(train_data, train_labels, train_labels_hot,
 val_data, val_labels, val_labels_hot,
 test_data, test_labels, test_labels_hot,
 train_ids, val_ids, test_ids,
 tot_train_val_idxs, tot_train_idxs, tot_val_idxs, tot_test_idxs) = DT.data_split_random(
    input_arr, unwanted_labels, unwanted_onehot, ids)

train_bios, val_bios, test_bios = diseases[tot_train_idxs], diseases[tot_val_idxs], diseases[tot_test_idxs]
bio_label2bat = {t: t for t in set(diseases)}

db_name = "Purity_lb{}".format(lb_str)
out_dir = os.path.join("DA_model/batch", "purity_lb{}/".format(lb_str))
os.makedirs(out_dir, exist_ok=True)

train_list = [train_data, train_labels, train_bios, train_ids, train_labels_hot]
val_list   = [val_data, val_labels, val_bios, val_ids, val_labels_hot]
test_list  = [test_data, test_labels, test_bios, test_ids, test_labels_hot]

# ---- train ----
trainer = RUN.train(train_list=train_list, val_list=val_list, test_list=test_list,
                    label2unw=label2bat, label2wnt=bio_label2bat,
                    net_args=net_args, out_dir=out_dir)

# ---- evaluate ----
ae_path = os.path.join(out_dir, "ae.tar")
if os.path.exists(ae_path):
    trainer.load_trained_ae(ae_path)
    print("loaded best ae.tar", flush=True)
else:
    print("WARNING: no ae.tar saved (need epochs > 450)", flush=True)

test_trans = TransData(test_data, test_labels, test_bios, test_ids, test_labels_hot)
test_loader = DataLoader(test_trans, batch_size=net_args.batch_size,
                         collate_fn=test_trans.collate_fn, shuffle=False, drop_last=False)
record_path = os.path.join(out_dir, "test_res.csv")
trainer.evaluate(record_path, db_name, test_loader)
print("DONE_TRAIN_EVAL", flush=True)
