#!/usr/bin/env python3
import os, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.decomposition import PCA
import faiss
from sklearn.metrics import roc_auc_score, f1_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR  = '/media/T2510596/HDD/thesis-mamba-nids/data/data_6feat'
CKPT_DIR  = '/media/T2510596/HDD/thesis-mamba-nids/checkpoints'
BASE_CKPT = os.path.join(CKPT_DIR, 'mamba_champion_hybrid.pth')
NEW_CKPT  = os.path.join(CKPT_DIR, 'mamba_uda_cicids_3ep.pth')

D_MODEL = 256; N_LAYERS = 2; D_STATE = 16; D_CONV = 4; EXPAND = 2
BATCH_SIZE = 512; LR = 1e-5; EPOCHS = 3; TAU = 0.5; PROJ_DIM = 128
CUTMIX_RATIO = 0.4; JITTER_STD = 0.15

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = np.array([d['features'] if isinstance(d, dict) else d for d in data], dtype=np.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx]

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

class PurePyTorchMamba(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.d_inner = d_model * EXPAND
        self.d_state = D_STATE
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, D_CONV, padding=D_CONV-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, D_STATE * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, D_STATE+1).float().unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    def forward(self, x):
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        x_conv = self.conv1d(x_in.transpose(1,2))[:,:,:l].transpose(1,2)
        x_conv = F.silu(x_conv)
        x_ssm = self.x_proj(x_conv)
        B = x_ssm[:,:,:self.d_state]
        C = x_ssm[:,:,self.d_state:2*self.d_state]
        dt = F.softplus(self.dt_proj(x_ssm[:,:,-1:]))
        A = -torch.exp(self.A_log)
        y_list = []
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device)
        for t in range(l):
            dt_t = dt[:,t,:].unsqueeze(-1)
            h = torch.exp(A.unsqueeze(0)*dt_t) * h + dt_t * B[:,t,:].unsqueeze(1) * x_conv[:,t,:].unsqueeze(-1)
            y_list.append((h * C[:,t,:].unsqueeze(1)).sum(-1))
        y = torch.stack(y_list, dim=1)
        y = y * F.silu(z) + x_in * self.D.unsqueeze(0).unsqueeze(0)
        return self.out_proj(y)

class MambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.embedder = MixedPacketEmbedder6(d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([PurePyTorchMamba(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.embedder(x); h = self.dropout(h)
        for layer in self.layers: h = layer(h) + h
        return self.norm(h)
    def encode_pooled(self, x):
        return self.forward(x).mean(dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, d_in, d_out=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(), nn.Linear(d_in, d_out))
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)

def nt_xent_loss(z1, z2, tau=0.5):
    z = torch.cat([z1, z2], dim=0)
    N = z1.size(0)
    sim = torch.mm(z, z.t()) / tau
    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, -1e9)
    pos = torch.cat([torch.diag(sim, N), torch.diag(sim, -N)])
    return -pos.mean() + torch.logsumexp(sim, dim=1).mean()

def dual_augment(batch):
    B, T, F_dim = batch.shape
    n_cut = int(T * CUTMIX_RATIO)
    views = []
    for _ in range(2):
        idx = torch.randperm(B, device=batch.device)
        donor = batch[idx]
        pos = torch.randint(0, T - n_cut + 1, (B,), device=batch.device)
        v = batch.clone()
        for i in range(B):
            v[i, pos[i]:pos[i]+n_cut] = donor[i, pos[i]:pos[i]+n_cut]
        jitter_len = 1.0 + torch.randn(B, T, 1, device=batch.device) * JITTER_STD
        jitter_iat = 1.0 + torch.randn(B, T, 1, device=batch.device) * JITTER_STD
        v[:, :, 1:2] = v[:, :, 1:2] * jitter_len
        v[:, :, 3:4] = v[:, :, 3:4] * jitter_iat
        views.append(v)
    return views[0], views[1]

print("="*70)
print("  UDA FINE-TUNING — 3 EPOCHS")
print("="*70)

print(f"Loading {BASE_CKPT}...")
encoder = MambaEncoder(D_MODEL, N_LAYERS).to(DEVICE)
proj = ProjectionHead(D_MODEL, PROJ_DIM).to(DEVICE)
ckpt = torch.load(BASE_CKPT, map_location=DEVICE, weights_only=False)
encoder.load_state_dict(ckpt['encoder'])
# Load proj head if it exists, otherwise train from scratch
if 'proj' in ckpt:
    proj.load_state_dict(ckpt['proj'])

print("Loading CICIDS Target Domain Data...")
with open(os.path.join(DATA_DIR, 'cicids_6feat_eval_v2.pkl'), 'rb') as f:
    data = pickle.load(f)

# Split out attacks and benign
labels = np.array([d.get('label', 0) if isinstance(d, dict) else 0 for d in data])
benign_data = [data[i] for i in range(len(data)) if labels[i] == 0]
attack_data = [data[i] for i in range(len(data)) if labels[i] == 1]
del data

# 80/20 train/test split on benign data
train_benign, test_benign = train_test_split(benign_data, train_size=0.8, random_state=42)
print(f"Train Benign: {len(train_benign):,} | Test Benign: {len(test_benign):,} | Attacks: {len(attack_data):,}")

# Reduce RAM
del benign_data

train_loader = DataLoader(FlowDataset(train_benign), batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, drop_last=True) # num_workers=0 to prevent RAM dup

opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=LR, weight_decay=1e-5)

print("\nStarting UDA Fine-Tuning...")
for epoch in range(EPOCHS):
    encoder.train(); proj.train()
    total_loss, n_batches = 0, 0
    t0 = time.time()
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = batch.to(DEVICE)
        v1, v2 = dual_augment(batch)
        z1 = proj(encoder.encode_pooled(v1))
        z2 = proj(encoder.encode_pooled(v2))
        loss = nt_xent_loss(z1, z2, TAU)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item()
        n_batches += 1
        
    avg_train = total_loss / max(1, n_batches)
    elapsed = time.time() - t0
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train:.4f} | {elapsed:.0f}s")

print(f"\nSaving UDA Checkpoint to {NEW_CKPT}...")
torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict(), 'epoch': EPOCHS}, NEW_CKPT)

# Extract embeddings for Evaluation
encoder.eval()
print("\nExtracting Train Embeddings (for PCA/FAISS Reference Pool)...")
train_feats = np.array([d['features'] if isinstance(d,dict) else d for d in train_benign], dtype=np.float32)
train_emb = []
with torch.no_grad():
    for i in tqdm(range(0, len(train_feats), BATCH_SIZE)):
        bt = torch.from_numpy(train_feats[i:i+BATCH_SIZE]).to(DEVICE)
        train_emb.append(encoder.encode_pooled(bt).cpu().numpy())
train_emb = np.concatenate(train_emb, axis=0)

print("\nExtracting Test Embeddings (Benign + Attacks)...")
test_data = test_benign + attack_data
test_labels = np.array([0] * len(test_benign) + [1] * len(attack_data))
test_feats = np.array([d['features'] if isinstance(d,dict) else d for d in test_data], dtype=np.float32)

test_emb = []
with torch.no_grad():
    for i in tqdm(range(0, len(test_feats), BATCH_SIZE)):
        bt = torch.from_numpy(test_feats[i:i+BATCH_SIZE]).to(DEVICE)
        test_emb.append(encoder.encode_pooled(bt).cpu().numpy())
test_emb = np.concatenate(test_emb, axis=0)

del train_feats, test_feats, test_data, train_benign, test_benign, attack_data

print("\nFitting PCA-12D and FAISS on Fine-Tuned Domain...")
pca = PCA(n_components=12, whiten=True, random_state=42)
# UDA fits PCA on its own fine-tuned training target domain (up to 200,000 to save memory)
tp = pca.fit_transform(train_emb[:200000]).astype(np.float32)
ep = pca.transform(test_emb).astype(np.float32)

faiss.normalize_L2(tp); faiss.normalize_L2(ep)

res = faiss.StandardGpuResources()
idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(12))
idx.add(tp)

print("\n--- UDA CROSS-DOMAIN EVALUATION ---")
test_sims, _ = idx.search(ep, 1)
test_scores = 1.0 - test_sims[:,0]

auc = roc_auc_score(test_labels, test_scores)
print(f"UDA Fine-Tuned ROC-AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(test_labels, test_scores)
best_idx = np.argmax(tpr - fpr)
y_threshold = thresholds[best_idx]
y_preds = (test_scores > y_threshold).astype(int)

y_mf1 = f1_score(test_labels, y_preds, average='macro')
y_rec = recall_score(test_labels, y_preds)
y_fp = ((y_preds == 1) & (test_labels == 0)).sum()
y_tn = ((y_preds == 0) & (test_labels == 0)).sum()
y_far = (y_fp / (y_fp + y_tn)) * 100 if (y_fp+y_tn) > 0 else 0

print("-" * 60)
print(f"{'Youden J Threshold':<18} | {'Macro F1':<10} | {'Recall':<10} | {'FAR (%)':<10}")
print(f"{y_threshold:<18.6f} | {y_mf1:<10.4f} | {y_rec:<10.4f} | {y_far:<10.4f}%")
print("=" * 60)
