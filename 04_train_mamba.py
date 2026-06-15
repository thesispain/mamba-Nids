#!/usr/bin/env python3
import os, sys, time, pickle, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_DIR = os.path.join(BASE_DIR, 'checkpoints')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

D_MODEL = 256; N_LAYERS = 4; D_STATE = 16; EXPAND = 2; D_CONV = 4
BATCH_SIZE = 512; LR = 1e-4; EPOCHS = 100; TAU = 0.5; PROJ_DIM = 128
PATIENCE = 8; CUTMIX_RATIO = 0.4; JITTER_STD = 0.15

class FlowDataset(Dataset):
    def __init__(self, data):
        self.features = np.array([d['features'] for d in data], dtype=np.float32)
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
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                padding=d_conv-1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state+1).float().unsqueeze(0).expand(self.d_inner, -1)
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
            A_bar = torch.exp(A.unsqueeze(0) * dt_t)
            B_bar = dt_t * B[:,t,:].unsqueeze(1)
            h = A_bar * h + B_bar * x_conv[:,t,:].unsqueeze(-1)
            y_t = (h * C[:,t,:].unsqueeze(1)).sum(-1)
            y_list.append(y_t)
        y = torch.stack(y_list, dim=1)
        y = y * F.silu(z) + x_in * self.D.unsqueeze(0).unsqueeze(0)
        return self.out_proj(y)

class MambaEncoder(nn.Module):
    def __init__(self, d_model=256, n_layers=2):
        super().__init__()
        self.embedder = MixedPacketEmbedder6(d_model)
        self.dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([PurePyTorchMamba(d_model, D_STATE, D_CONV, EXPAND) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.embedder(x)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h) + h
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

def train():
    print(f"Device: {DEVICE}")

    with open(os.path.join(DATA_DIR, 'champion_train.pkl'), 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data):,} benign flows")

    np.random.seed(42)
    idx = np.random.permutation(len(data))
    split = int(len(data) * 0.9)
    train_data = [data[i] for i in idx[:split]]
    val_data   = [data[i] for i in idx[split:]]
    del data

    train_loader = DataLoader(FlowDataset(train_data), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(FlowDataset(val_data), batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    encoder = MambaEncoder(D_MODEL, N_LAYERS).to(DEVICE)
    proj = ProjectionHead(D_MODEL, PROJ_DIM).to(DEVICE)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(proj.parameters()), lr=LR, weight_decay=1e-5)

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {total_params:,}")

    best_val = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(EPOCHS):
        encoder.train(); proj.train()
        total_loss, n_batches = 0, 0
        t0 = time.time()
        for batch in train_loader:
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
            if n_batches % 500 == 0:
                print(f"  Epoch [{epoch+1}/{EPOCHS}] Step [{n_batches}/{len(train_loader)}] Loss: {loss.item():.4f}", flush=True)
        avg_train = total_loss / max(1, n_batches)

        encoder.eval(); proj.eval()
        val_loss, vn = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                v1, v2 = dual_augment(batch)
                z1 = proj(encoder.encode_pooled(v1))
                z2 = proj(encoder.encode_pooled(v2))
                val_loss += nt_xent_loss(z1, z2, TAU).item()
                vn += 1
        avg_val = val_loss / max(1, vn)
        elapsed = time.time() - t0
        history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | {elapsed:.0f}s", flush=True)

        if avg_val < best_val:
            best_val = avg_val
            patience_counter = 0
            torch.save({'encoder': encoder.state_dict(), 'proj': proj.state_dict(),
                        'epoch': epoch+1, 'val_loss': avg_val},
                       os.path.join(CKPT_DIR, 'mamba_champion_absolute.pt'))
            print(f"  Checkpoint saved. Best val: {best_val:.4f}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(RESULTS_DIR, 'champion_training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\nExtracting embeddings with best checkpoint...")
    ckpt = torch.load(os.path.join(CKPT_DIR, 'mamba_champion_absolute.pt'), map_location=DEVICE, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()

    for name, pkl_name, emb_name in [
        ('Train', 'champion_train.pkl', 'champion_abs_train_emb.npy'),
        ('Eval',  'champion_eval.pkl',  'champion_abs_eval_emb.npy')
    ]:
        print(f"  {name}...")
        with open(os.path.join(DATA_DIR, pkl_name), 'rb') as f:
            split_data = pickle.load(f)
        feats = np.array([d['features'] for d in split_data], dtype=np.float32)
        all_emb = []
        with torch.no_grad():
            for i in range(0, len(feats), BATCH_SIZE):
                bt = torch.from_numpy(feats[i:i+BATCH_SIZE]).to(DEVICE)
                all_emb.append(encoder.encode_pooled(bt).cpu().numpy())
        all_emb = np.concatenate(all_emb, axis=0)
        np.save(os.path.join(DATA_DIR, emb_name), all_emb)
        print(f"    Saved: {emb_name} — {all_emb.shape}")
        del split_data, feats

    print(f"\nTraining complete. Best val: {best_val:.4f}")

if __name__ == '__main__':
    train()
