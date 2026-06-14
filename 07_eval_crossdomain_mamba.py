#!/usr/bin/env python3
import time, os, sys, torch, pickle, faiss, gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

try:
    from mamba_ssm import Mamba
except ImportError:
    print("mamba_ssm not found. Please install it first.")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'mamba_crossdomain_v1_latest.pt')
D_MODEL = 256; N_LAYERS = 2; D_STATE = 16; EXPAND = 2; D_CONV = 4

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

encoder = MambaEncoder(D_MODEL, N_LAYERS).to(DEVICE)
print(f"Loading checkpoint from: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
if 'encoder' in ckpt: encoder.load_state_dict(ckpt['encoder'])
else: encoder.load_state_dict(ckpt)
encoder.eval()

def extract_embeddings(data_list, desc="Data"):
    print(f"Extracting {desc}...", flush=True)
    loader = DataLoader(FlowDataset([flatten_stats(d) for d in data_list]), batch_size=512, shuffle=False)
    emb_list = []
    with torch.no_grad():
        for xb in loader:
            emb_list.append(encoder.encode_pooled(xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(emb_list)

# ── 1. Load Source Domain Train (UNSW) for PCA Fitting ────────────────────────
print("\nLoading UNSW Train for PCA fit...")
with open(os.path.join(DATA_DIR, 'unsw_hybrid_train.pkl'), 'rb') as f: train_data = pickle.load(f)
source_train_emb = extract_embeddings(train_data, "Source Train Embeddings")
del train_data; gc.collect()

# ── 2. Load Target Domain (CICIDS-17) for Evaluation ─────────────────────────
print("\nLoading CICIDS-2017 Eval Data...")
with open(os.path.join(DATA_DIR, 'cicids_6feat_eval_v2.pkl'), 'rb') as f: eval_data = pickle.load(f)

labels = np.array([d.get('label', 0) if isinstance(d, dict) else 0 for d in eval_data])
benign_idx = np.where(labels == 0)[0]
attack_idx = np.where(labels == 1)[0]

# Split 70/10/5/15 like Honest Validation
# We only need 10% for Ref, 15% Test for Benign
b_train, b_rest = train_test_split(benign_idx, train_size=0.70, random_state=42)
b_ref, b_test_val = train_test_split(b_rest, train_size=0.3333, random_state=42) # 33% of 30% = 10%
b_val, b_test = train_test_split(b_test_val, train_size=0.25, random_state=42) # 25% of 20% = 5%, leaving 15% test

# Attacks: 20% Val, 80% Test
a_val, a_test = train_test_split(attack_idx, train_size=0.20, random_state=42)

# Build Eval Arrays
ref_indices = np.sort(b_ref)
test_indices = np.sort(np.concatenate([b_test, a_test]))

print("Extracting Target Domain Embeddings...")
ref_data = [eval_data[i] for i in ref_indices]
test_data = [eval_data[i] for i in test_indices]
test_labels = labels[test_indices]
del eval_data; gc.collect()

ref_emb = extract_embeddings(ref_data, "Reference Pool Embeddings")
test_emb = extract_embeddings(test_data, "Test Set Embeddings")
del ref_data, test_data; gc.collect()

# ── 3. FAISS Evaluation Logic ────────────────────────────────────────────────
print("\n" + "═"*80)
print("  [3/3] HONEST CROSS-DOMAIN FAISS EVALUATION")
print("═"*80)

configs = [('PCA-12D+k=1', 12, 1)] 
best_row = None 

for name, pd, k in configs: 
    pca = PCA(n_components=pd, whiten=True, random_state=42) 
    tp = pca.fit_transform(source_train_emb).astype(np.float32)   
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

    final_threshold = np.percentile(ref_scores, best_pct)

    # Score Test Set
    test_sims, _ = idx.search(ep, k)
    test_scores = 1.0 - (test_sims[:,0] if k==1 else test_sims.mean(axis=1))
    preds = (test_scores > final_threshold).astype(int)

    auc = roc_auc_score(test_labels, test_scores)
    ap  = average_precision_score(test_labels, test_scores)
    mf1 = f1_score(test_labels, preds, average='macro', zero_division=0)
    fp_count = ((preds==1)&(test_labels==0)).sum()
    tn_count = ((preds==0)&(test_labels==0)).sum()
    far = fp_count/(fp_count+tn_count) if (fp_count+tn_count)>0 else 0

    print(f"  {name:<16} | AUC: {auc:>8.4f} | F1: {mf1:>8.4f} | FAR: {far:>7.4%} | Threshold Pct: {best_pct}")

print("\n" + "="*80)
print(f"  FINAL MAMBA CROSS-DOMAIN METRICS (UNSW -> CICIDS17)")
print("="*80)
print(f"  ROC-AUC   : {auc:.4f}")
print(f"  PR-AUC    : {ap:.4f}")
print(f"  Macro F1  : {mf1:.4f}")
print(f"  FAR       : {far:.4%}")
print("="*80)
