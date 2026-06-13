#!/usr/bin/env python3
import time, os, sys, torch, pickle, faiss, gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score)

try:
    from mamba_ssm import Mamba
except ImportError:
    print("mamba_ssm not found. Please install it first.")
    sys.exit(1)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

DATA_DIR  = '/media/T2510596/HDD/thesis-mamba-nids/data/data_6feat'
CKPT_PATH = '/media/T2510596/HDD/thesis-mamba-nids/checkpoints/mamba_crossdomain_v1_latest.pt'
D_MODEL = 256; N_LAYERS = 2; D_STATE = 16; EXPAND = 2; D_CONV = 4

# ── Model Architecture (Identical to 11_retrain_mamba_cd.py) ───────────────
class MixedPacketEmbedder6(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.proto_embed = nn.Embedding(256, 48)
        self.len_proj = nn.Linear(1, 40)
        self.flags_embed = nn.Embedding(256, 48)
        self.iat_proj = nn.Linear(1, 40)
        self.dir_embed = nn.Embedding(3, 32)
        self.port_cat_embed = nn.Embedding(3, 48)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        proto = self.proto_embed(x[:,:,0].long().clamp(0,255))
        length = self.len_proj(x[:,:,1:2])
        flags = self.flags_embed(x[:,:,2].long().clamp(0,255))
        iat = self.iat_proj(x[:,:,3:4])
        direction = self.dir_embed(x[:,:,4].long().clamp(0,2))
        port_cat = self.port_cat_embed(x[:,:,5].long().clamp(0,2))
        return self.norm(torch.cat([proto, length, flags, iat, direction, port_cat], dim=-1))

class MambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.embedder = MixedPacketEmbedder6(d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([
            Mamba(d_model=d_model, d_state=D_STATE, d_conv=D_CONV, expand=EXPAND) 
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.embedder(x); h = self.dropout(h)
        for layer in self.layers: h = layer(h) + h
        return self.norm(h)
    def encode_pooled(self, x):
        return self.forward(x).mean(dim=1)

def flatten_stats(d):
    feat = d['features'] if isinstance(d, dict) else d
    feat = np.array(feat)[:, :6]
    T = feat.shape[0]
    if T >= 32: feat = feat[:32]
    else: feat = np.vstack([feat, np.zeros((32 - T, 6))])
    return feat.astype(np.float32)

class FlowDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return torch.tensor(self.data[idx])

# ── 1. Params & FLOPs ───────────────────────────────────────────────────────
print("═"*80)
print("  [1/3] HARDWARE & COMPLEXITY METRICS")
print("═"*80)
encoder = MambaEncoder(D_MODEL, N_LAYERS).to(DEVICE)

# Load killed script weights
print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
if 'encoder' in ckpt:
    encoder.load_state_dict(ckpt['encoder'])
else:
    encoder.load_state_dict(ckpt)
encoder.eval()

param_count = sum(p.numel() for p in encoder.parameters())
print(f"  Total Parameters: {param_count:,}")

# Try to estimate FLOPs
try:
    from thop import profile
    dummy_input = torch.zeros(1, 32, 6, device=DEVICE)
    macs, _ = profile(encoder, inputs=(dummy_input,), verbose=False)
    print(f"  Estimated MACs: {macs:,} ({macs/1e6:.2f} M)")
except ImportError:
    print("  thop not installed, skipping MACs calculation.")
except Exception as e:
    print(f"  thop MACs calculation failed (common for custom CUDA kernels): {e}")

# ── 2. Extracing Embeddings Dynamically ──────────────────────────────────────
print("\n" + "═"*80)
print("  [2/3] DYNAMIC EMBEDDING EXTRACTION")
print("═"*80)

def extract_embeddings(data_list, desc="Data"):
    print(f"Extracting {desc}...", flush=True)
    loader = DataLoader(FlowDataset([flatten_stats(d) for d in data_list]), batch_size=512, shuffle=False)
    emb_list = []
    with torch.no_grad():
        for xb in loader:
            emb_list.append(encoder.encode_pooled(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(emb_list)

print("Loading UNSW Hybrid Train...")
with open(os.path.join(DATA_DIR, 'unsw_hybrid_train.pkl'), 'rb') as f: train_data = pickle.load(f)
train_emb = extract_embeddings(train_data, "Train Embeddings")
del train_data; gc.collect()

print("Loading UNSW Hybrid Eval...")
with open(os.path.join(DATA_DIR, 'unsw_hybrid_eval.pkl'), 'rb') as f: eval_data = pickle.load(f)
labels = np.array([d['label'] for d in eval_data])
eval_emb = extract_embeddings(eval_data, "Eval Embeddings")
del eval_data; gc.collect()

# ── 3. FAISS Evaluation Logic ────────────────────────────────────────────────
print("\n" + "═"*80)
print("  [3/3] HONEST FAISS GAP-TUNING EVALUATION")
print("═"*80)

benign_indices = np.where(labels == 0)[0]
attack_indices = np.where(labels == 1)[0]
ref_indices, test_benign_indices = train_test_split(benign_indices, test_size=0.85, random_state=42)
test_indices = np.sort(np.concatenate([test_benign_indices, attack_indices]))

ref_emb = eval_emb[ref_indices]
test_emb = eval_emb[test_indices]
test_labels = labels[test_indices]

# Free memory before FAISS
del eval_emb; gc.collect()

configs = [('PCA-12D+k=1', 12, 1)] # Single best config to save time
best_row = None 

print(f"\nRunning HONEST GAP TUNING on Reference Pool...")
print(f"  {'Config':<16} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Macro F1':>8} | {'FAR':>8} | {'Best Pct':>10}")
print("  " + "-"*80)

for name, pd, k in configs: 
    pca = PCA(n_components=pd, whiten=True, random_state=42) 
    tp = pca.fit_transform(train_emb).astype(np.float32)   
    rp = pca.transform(ref_emb).astype(np.float32)
    ep = pca.transform(test_emb).astype(np.float32)          
    faiss.normalize_L2(tp); faiss.normalize_L2(rp); faiss.normalize_L2(ep)

    res = faiss.StandardGpuResources() 
    idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(pd)) 
    idx.add(tp) 

    # Score Reference Pool
    ref_sims, _ = idx.search(rp, k)
    ref_scores = 1.0 - (ref_sims[:,0] if k==1 else ref_sims.mean(axis=1))

    # HONEST GAP SWEEP (on Reference Pool ONLY)
    best_pct = 97.0
    best_gap = -1
    for pct in [90, 91, 92, 93, 94, 95, 96, 97, 97.5, 98, 98.5, 99, 99.5]:
        t = np.percentile(ref_scores, pct)
        above = ref_scores[ref_scores > t]
        below = ref_scores[ref_scores <= t]
        if len(above) > 0 and len(below) > 0:
            gap = above.mean() - below.mean()
            if gap > best_gap:
                best_gap = gap
                best_pct = pct

    # Lock threshold
    final_threshold = np.percentile(ref_scores, best_pct)

    # Score Test Set
    test_sims, _ = idx.search(ep, k)
    test_scores = 1.0 - (test_sims[:,0] if k==1 else test_sims.mean(axis=1))
    preds = (test_scores > final_threshold).astype(int)

    # Metrics
    auc = roc_auc_score(test_labels, test_scores)
    ap  = average_precision_score(test_labels, test_scores)
    mf1 = f1_score(test_labels, preds, average='macro', zero_division=0)
    fp_count = ((preds==1)&(test_labels==0)).sum()
    tn_count = ((preds==0)&(test_labels==0)).sum()
    far = fp_count/(fp_count+tn_count) if (fp_count+tn_count)>0 else 0

    print(f"  {name:<16} | {auc:>8.4f} | {ap:>8.4f} | {mf1:>8.4f} | {far:>7.4%} | {best_pct:>10}")

    if best_row is None or mf1 > best_row['f_macro']:
        best_row = {'name':name, 'f_macro':mf1, 'pct':best_pct, 'far':far, 'auc':auc, 'threshold':final_threshold, 'pr':ap}

print("\n" + "="*80)
print(f"  FINAL MAMBA UNSW IN-DOMAIN METRICS (EPOCH 40 CHECKPOINT)")
print("="*80)
print(f"  ROC-AUC   : {best_row['auc']:.4f}")
print(f"  PR-AUC    : {best_row['pr']:.4f}")
print(f"  Macro F1  : {best_row['f_macro']:.4f}")
print(f"  FAR       : {best_row['far']:.4%}")
print(f"  Parameters: {param_count:,}")
print("="*80)
