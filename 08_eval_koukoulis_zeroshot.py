#!/usr/bin/env python3
import time, os, sys, torch, pickle, faiss, gc
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

try:
    from mamba_ssm import Mamba
except ImportError:
    print("mamba_ssm not found.")
    sys.exit(1)

DATA_DIR  = '/media/T2510596/HDD/thesis-mamba-nids/data/data_6feat'
CKPT_PATH = '/media/T2510596/HDD/thesis-mamba-nids/checkpoints/mamba_unsw_4layer_30ep.pt'
D_MODEL = 256; N_LAYERS = 4; D_STATE = 16; EXPAND = 2; D_CONV = 4

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

print("\nLoading UNSW Train Data (Source Domain)...")
with open(os.path.join(DATA_DIR, 'unsw_hybrid_train.pkl'), 'rb') as f: source_data = pickle.load(f)
source_train_emb = extract_embeddings(source_data, "UNSW Train Embeddings")
del source_data; gc.collect()

print("\nLoading CICIDS Eval Data (Target Domain)...")
with open(os.path.join(DATA_DIR, 'cicids_6feat_eval_v2.pkl'), 'rb') as f: target_data = pickle.load(f)
target_labels = np.array([d.get('label', 0) if isinstance(d, dict) else 0 for d in target_data])
target_eval_emb = extract_embeddings(target_data, "CICIDS Eval Embeddings")
del target_data; gc.collect()

print("\n" + "═"*80)
print("  STRICT KOUKOULIS ZERO-SHOT EVALUATION (Reference = UNSW Train)")
print("═"*80)

# Normalize
faiss.normalize_L2(source_train_emb)
faiss.normalize_L2(target_eval_emb)

res = faiss.StandardGpuResources() 
idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(D_MODEL)) 
idx.add(source_train_emb) 

# Search K=1 against the entire UNSW training set
print("Searching for nearest neighbors in UNSW Train for every CICIDS packet...")
sims, _ = idx.search(target_eval_emb, 1)
test_scores = 1.0 - sims[:, 0]

auc = roc_auc_score(target_labels, test_scores)

print(f"\nSTRICT ZERO-SHOT ROC-AUC (CICIDS using UNSW Reference Pool): {auc:.4f}")
print("═"*80)
