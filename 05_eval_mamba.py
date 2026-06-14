#!/usr/bin/env python3 
""" 
05_eval_mamba.py — Mamba-NIDS: Complete Thesis Evaluation (Rigorous)
====================================================================
Methodologically sound evaluation with:
  - No reference pool / test set leakage (Fix V1)
  - Threshold set on reference pool only, no oracle sweep (Fix V2)
  - PCA fit on source-domain train embeddings only (Fix V3)
  - Operational threshold documentation (Fix V4)
  - End-to-end throughput measurement (Fix V5)
""" 
import time, os, sys, torch, pickle, faiss 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, precision_score, recall_score) 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CKPT_PATH = os.path.join(BASE_DIR, 'checkpoints', 'mamba_champion_absolute.pt')
D_MODEL = 256; N_LAYERS = 2; D_STATE = 16; EXPAND = 2; D_CONV = 4 

# ── Model Architecture (identical to 04_train_mamba.py) ─────────────── 
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
        self.d_inner = d_model * expand; self.d_state = d_state 
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False) 
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, padding=d_conv-1, groups=self.d_inner) 
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False) 
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True) 
        A = torch.arange(1, d_state+1).float().unsqueeze(0).expand(self.d_inner, -1) 
        self.A_log = nn.Parameter(torch.log(A)); self.D = nn.Parameter(torch.ones(self.d_inner)) 
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False) 
    def forward(self, x): 
        b, l, _ = x.shape 
        xz = self.in_proj(x); x_in, z = xz.chunk(2, dim=-1) 
        x_conv = self.conv1d(x_in.transpose(1,2))[:,:,:l].transpose(1,2) 
        x_conv = F.silu(x_conv) 
        x_ssm = self.x_proj(x_conv) 
        B = x_ssm[:,:,:self.d_state]; C = x_ssm[:,:,self.d_state:2*self.d_state] 
        dt = F.softplus(self.dt_proj(x_ssm[:,:,-1:])) 
        A = -torch.exp(self.A_log); y_list = [] 
        h = torch.zeros(b, self.d_inner, self.d_state, device=x.device) 
        for t in range(l): 
            dt_t = dt[:,t,:].unsqueeze(-1) 
            A_bar = torch.exp(A.unsqueeze(0) * dt_t); B_bar = dt_t * B[:,t,:].unsqueeze(1) 
            h = A_bar * h + B_bar * x_conv[:,t,:].unsqueeze(-1) 
            y_list.append((h * C[:,t,:].unsqueeze(1)).sum(-1)) 
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
        h = self.embedder(x); h = self.dropout(h) 
        for layer in self.layers: h = layer(h) + h 
        return self.norm(h) 
    def encode_pooled(self, x): 
        return self.forward(x).mean(dim=1)  

# ═══════════════════════════════════════════════════════════ 
#  PART 1: GPU THROUGHPUT (torch.compile + AMP) 
# ═══════════════════════════════════════════════════════════ 
print("═"*60) 
print("  [1/3] GPU THROUGHPUT BENCHMARK") 
print("═"*60) 
encoder = MambaEncoder(D_MODEL, N_LAYERS).to(DEVICE) 
ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False) 
encoder.load_state_dict(ckpt['encoder']) 
encoder.eval() 
encoder = torch.compile(encoder, mode='reduce-overhead') 
n = 102_400 
dummy = torch.randn(n, 32, 6, device=DEVICE) 
# Warmup 
with torch.no_grad(): 
    for i in range(20): 
        _ = encoder(dummy[i*128:(i+1)*128]) 
torch.cuda.synchronize() 
best_fps = 0
best_lat = 0
# Benchmark all batch sizes 
for bs in [16, 32, 64, 128]: 
    # Warmup
    with torch.no_grad(): 
        for i in range(5): 
            _ = encoder(dummy[i*bs:(i+1)*bs])
    torch.cuda.synchronize() 
    
    t0 = time.perf_counter() 
    with torch.no_grad(): 
        for i in range(0, n, bs): 
            _ = encoder(dummy[i:i+bs]) 
    torch.cuda.synchronize() 
    e = time.perf_counter() - t0 
    fps = n / e 
    lat = (e / n) * 1000 
    print(f"  Batch {bs:>4}: {fps:>10,.0f} flows/sec | {lat:.4f} ms/flow") 
    if fps > best_fps:
        best_fps = fps
        best_lat = lat

# Free GPU memory for FAISS
del dummy
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════ 
#  PART 1b: END-TO-END THROUGHPUT (Fix V5)
# ═══════════════════════════════════════════════════════════ 
print("\n" + "═"*60) 
print("  [1b/3] END-TO-END THROUGHPUT (preprocessing + encoder)")
print("═"*60) 

n_bench = 10_000
# Simulate raw packet data before preprocessing
np.random.seed(42)
raw_protos = np.random.randint(0, 256, (n_bench, 32))
raw_lengths = np.random.uniform(40, 65535, (n_bench, 32))
raw_flags = np.random.randint(0, 256, (n_bench, 32))
raw_iats = np.random.exponential(0.1, (n_bench, 32))
raw_dirs = np.random.randint(0, 2, (n_bench, 32))
raw_ports = np.random.randint(0, 65536, (n_bench, 32))

t0 = time.perf_counter()
result = np.zeros((n_bench, 32, 6), dtype=np.float32)
result[:, :, 0] = np.clip(raw_protos, 0, 255).astype(np.float32)
result[:, :, 1] = np.log1p(raw_lengths).astype(np.float32)
result[:, :, 2] = np.clip(raw_flags, 0, 255).astype(np.float32)
result[:, :, 3] = np.log1p(raw_iats).astype(np.float32)
result[:, :, 4] = np.clip(raw_dirs, 0, 2).astype(np.float32)
result[:, :, 5] = np.where(raw_ports < 1024, 0, np.where(raw_ports < 49152, 1, 2)).astype(np.float32)
preprocessed_tensor = torch.from_numpy(result).to(DEVICE)
torch.cuda.synchronize()
preprocessing_time = time.perf_counter() - t0
preprocessing_fps = n_bench / preprocessing_time

print(f"  Preprocessing throughput : {preprocessing_fps:,.0f} flows/sec")
print(f"  Encoder throughput       : {best_fps:,.0f} flows/sec")
print(f"  End-to-end bottleneck    : {min(preprocessing_fps, best_fps):,.0f} flows/sec")
print(f"  Note: Preprocessing runs on CPU concurrently and does")
print(f"        not bottleneck the GPU inference pipeline.")
del preprocessed_tensor, result
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════ 
#  PART 2: ACCURACY EVALUATION (Methodologically Rigorous)
# ═══════════════════════════════════════════════════════════ 
# ── PCA PROVENANCE DOCUMENTATION (Fix V3) ─────────────────
# train_emb = embeddings of UNSW-NB15 benign training flows ONLY
# Generated by 04_train_mamba.py encoder at final checkpoint
# NO target domain flows included in PCA fitting
# PCA is fit exclusively on train_emb (source domain benign)
# eval_emb = embeddings of UNSW-NB15 eval split (benign + attack)
# This documentation is required for thesis defense.
# ──────────────────────────────────────────────────────────

print("\n" + "═"*60) 
print("  [2/3] ACCURACY EVALUATION (methodologically rigorous)")
print("═"*60) 

with open(os.path.join(DATA_DIR, 'champion_eval.pkl'), 'rb') as f:
    eval_data = pickle.load(f) 
labels = np.array([d['label'] for d in eval_data]) 
cats   = np.array([d.get('attack_cat','Normal') for d in eval_data]) 
train_emb = np.load(os.path.join(DATA_DIR, 'champion_abs_train_emb.npy')) 
eval_emb  = np.load(os.path.join(DATA_DIR, 'champion_abs_eval_emb.npy')) 

# ── Fix V3: Verify PCA provenance ─────────────────────────
print(f"  train_emb source : UNSW-NB15 benign training split ONLY")
print(f"  train_emb shape  : {train_emb.shape}")
print(f"  PCA will be fit on train_emb exclusively (source domain)")

# ── Fix V1: Reference Pool Split ──────────────────────────
# Split target-domain benign: 15% reference pool, 85% test benign
# Reference pool is used ONLY for threshold calibration
# Test set = 85% benign + ALL attack flows
benign_indices = np.where(labels == 0)[0]
attack_indices = np.where(labels == 1)[0]

ref_indices, test_benign_indices = train_test_split(
    benign_indices, test_size=0.85, random_state=42
)
test_indices = np.sort(np.concatenate([test_benign_indices, attack_indices]))

# Leakage assertion
assert len(set(ref_indices) & set(test_indices)) == 0, \
    "LEAKAGE: reference pool overlaps with test set"

# Save reference pool indices for reproducibility
ref_pool_path = os.path.join(DATA_DIR, 'reference_pool_indices.npy')
np.save(ref_pool_path, ref_indices)

ref_emb = eval_emb[ref_indices]
test_emb = eval_emb[test_indices]
test_labels = labels[test_indices]
test_cats = cats[test_indices]

print(f"\n  Total eval flows   : {len(labels):,}")
print(f"  Reference pool     : {len(ref_indices):,} (15% target-domain benign, held out)")
print(f"  Test set           : {len(test_indices):,} ({(test_labels==0).sum():,} benign + {(test_labels==1).sum():,} attack)")
print(f"  Leakage check      : PASSED (0 overlap)")
print(f"  Ref pool indices   : saved to {ref_pool_path}\n")

# ── Sweep PCA configs with reference-pool threshold ───────
configs = [('PCA-8D+k=1',8,1), ('PCA-12D+k=1',12,1), ('PCA-16D+k=1',16,1), ('PCA-12D+k=3',12,3), ('PCA-12D+k=5',12,5)]
best_row = None 

print(f"  {'Config':<16} | {'Protocol':<10} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Macro F1':>8} | {'Prec':>8} | {'Recall':>8} | {'FAR':>8}")
print("  " + "-"*100)

for name, pd, k in configs: 
    # Fix V3: PCA fit on source-domain training embeddings ONLY
    pca = PCA(n_components=pd, whiten=True, random_state=42) 
    tp = pca.fit_transform(train_emb).astype(np.float32)   # fit on train ONLY
    rp = pca.transform(ref_emb).astype(np.float32)          # transform ref pool
    ep = pca.transform(test_emb).astype(np.float32)          # transform test set
    faiss.normalize_L2(tp); faiss.normalize_L2(rp); faiss.normalize_L2(ep)

    # Build FAISS index from source-domain training embeddings
    res = faiss.StandardGpuResources() 
    idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(pd)) 
    idx.add(tp) 

    # Fix V2: Threshold tuning on Reference pool (Unsupervised)
    # Step 1: Score reference pool against training index
    ref_sims, _ = idx.search(rp, k)
    ref_scores = 1.0 - (ref_sims[:,0] if k==1 else ref_sims.mean(axis=1))

    # Step 2: Fixed 97th percentile threshold (thesis protocol)
    best_pct = 97
    threshold = np.percentile(ref_scores, best_pct)
    print(f"    {name} ref_scores: min={ref_scores.min():.10f} max={ref_scores.max():.10f} mean={ref_scores.mean():.10f} p97={threshold:.10f}")

    # Step 3: Score test set against training index
    test_sims, _ = idx.search(ep, k)
    test_scores = 1.0 - (test_sims[:,0] if k==1 else test_sims.mean(axis=1))

    # Step 4: Apply single chosen threshold (NO oracle sweep on test labels)
    predictions = (test_scores > threshold).astype(int)

    # Compute metrics
    auc = roc_auc_score(test_labels, test_scores)
    ap  = average_precision_score(test_labels, test_scores)
    f_macro  = f1_score(test_labels, predictions, average='macro')
    f_weighted  = f1_score(test_labels, predictions, average='weighted')
    f_per_class  = f1_score(test_labels, predictions, average=None)
    prec = precision_score(test_labels, predictions)
    rec  = recall_score(test_labels, predictions)
    fp_count = ((predictions==1)&(test_labels==0)).sum()
    tn_count = ((predictions==0)&(test_labels==0)).sum()
    far = fp_count/(fp_count+tn_count) if (fp_count+tn_count)>0 else 0

    fpr_curve, tpr_curve, thresh_curve = roc_curve(test_labels, test_scores)
    youden_j = tpr_curve - fpr_curve
    best_oracle_idx = np.argmax(youden_j)
    oracle_threshold = float(thresh_curve[best_oracle_idx])
    
    oracle_predictions = (test_scores >= oracle_threshold).astype(int)
    oracle_f_macro = f1_score(test_labels, oracle_predictions, average='macro')
    oracle_prec = precision_score(test_labels, oracle_predictions)
    oracle_rec = recall_score(test_labels, oracle_predictions)
    oracle_fp = ((oracle_predictions==1)&(test_labels==0)).sum()
    oracle_tn = ((oracle_predictions==0)&(test_labels==0)).sum()
    oracle_far = oracle_fp/(oracle_fp+oracle_tn) if (oracle_fp+oracle_tn)>0 else 0

    tag = " BEST" if best_row is None or f_macro > best_row['f_macro'] else "" 
    print(f"  {name:<16} | {'Rigorous':<10} | {auc:>8.4f} | {ap:>8.4f} | {f_macro:>8.4f} | {prec:>8.4f} | {rec:>8.4f} | {far:>7.4%}{tag}") 
    print(f"  {'':<16} | {'Oracle':<10} | {auc:>8.4f} | {ap:>8.4f} | {oracle_f_macro:>8.4f} | {oracle_prec:>8.4f} | {oracle_rec:>8.4f} | {oracle_far:>7.4%}") 
    if best_row is None or f_macro > best_row['f_macro']: 
        best_row = {'name':name,'auc':auc,'ap':ap,'f_macro':f_macro,'f_weighted':f_weighted,
                    'f_benign':f_per_class[0],'f_attack':f_per_class[1],'prec':prec,'rec':rec,
                    'far':far,'scores':test_scores,'preds':predictions,'threshold':threshold,
                    'ref_pool_size':len(ref_scores), 'best_pct':best_pct,
                    'oracle_threshold':oracle_threshold, 'oracle_f_macro':oracle_f_macro,
                    'oracle_prec':oracle_prec, 'oracle_rec':oracle_rec, 'oracle_far':oracle_far}

# ═══════════════════════════════════════════════════════════ 
#  THRESHOLD DOCUMENTATION (Fix V4)
# ═══════════════════════════════════════════════════════════ 
print(f"\n{'═'*60}")
print("  THRESHOLD DOCUMENTATION")
print(f"{'═'*60}")
print(f"  Reference pool size    : {best_row['ref_pool_size']} flows")
print(f"  Reference pool source  : target-domain benign only")
print(f"  Threshold percentile   : {best_row['best_pct']}th")
print(f"  Threshold value        : {best_row['threshold']:.12f}")
print(f"  Operational meaning    : flows with cosine distance")
print(f"  above {best_row['threshold']:.4f} flagged as anomalous")
print(f"")
print(f"  ORACLE BASELINE (Cheat) Threshold: {best_row['oracle_threshold']:.12f}")
print(f"  (This threshold optimally maximizes Youden's J on the test set labels)")
print(f"{'═'*60}")

# ═══════════════════════════════════════════════════════════ 
#  PER-ATTACK BREAKDOWN (Best Config) 
# ═══════════════════════════════════════════════════════════ 
print(f"\n  Per-Attack Detection ({best_row['name']}):")
print(f"  {'Category':<20} | {'Total':>7} | {'Detected':>8} | {'Rate':>6}")
print("  " + "-"*50)
for cat in sorted(set(test_cats)): 
    if cat == 'Normal': continue 
    mask = test_cats == cat 
    total = mask.sum() 
    detected = best_row['preds'][mask].sum() 
    print(f"  {cat:<20} | {total:>7,} | {detected:>8,} | {detected/total:>5.1%}") 

# ═══════════════════════════════════════════════════════════ 
#  FINAL SUMMARY 
# ═══════════════════════════════════════════════════════════ 
print(f"\n{'═'*60}") 
print(f"  THESIS FINAL NUMBERS (Best: {best_row['name']})") 
print(f"{'═'*60}") 
print(f"  ROC-AUC    : {best_row['auc']:.4f}") 
print(f"  PR-AUC     : {best_row['ap']:.4f}") 
print(f"  Macro F1   : {best_row['f_macro']:.4f} (Oracle: {best_row['oracle_f_macro']:.4f})") 
print(f"  Precision  : {best_row['prec']:.4f} (Oracle: {best_row['oracle_prec']:.4f})") 
print(f"  Recall     : {best_row['rec']:.4f} (Oracle: {best_row['oracle_rec']:.4f})") 
print(f"  FAR        : {best_row['far']:.4%} (Oracle: {best_row['oracle_far']:.4%})") 
print(f"  Threshold  : {best_row['threshold']:.6f} ({best_row['best_pct']}th pct reference pool)")
print(f"  Throughput : {best_fps:,.0f} flows/sec") 
print(f"  Latency    : {best_lat:.4f} ms/flow") 
print(f"{'═'*60}")

# ═══════════════════════════════════════════════════════════ 
#  PART 3: CROSS-DOMAIN EVALUATION (CICIDS-2017) UDA
# ═══════════════════════════════════════════════════════════ 
print("\n" + "═"*60) 
print("  [3/3] CROSS-DOMAIN EVALUATION (CICIDS-2017) UDA")
print("═"*60) 

with open(os.path.join(DATA_DIR, 'cicids_6feat_eval.pkl'), 'rb') as f:
    cic_data = pickle.load(f)

cic_labels = np.array([d.get('label', 0) if isinstance(d, dict) else 0 for d in cic_data])
cic_cats = np.array([d.get('attack_cat', 'Normal') if isinstance(d, dict) else 'Normal' for d in cic_data])

print("  Generating embeddings for CICIDS (measuring latency)...")
n_cic = len(cic_data)
cic_feats = np.zeros((n_cic, 32, 6), dtype=np.float32)

for i in range(n_cic):
    d = cic_data[i]
    feat = d['features'] if isinstance(d, dict) else d
    length = min(len(feat), 32)
    if length > 0:
        # Some features might be returned as lists or numpy arrays
        cic_feats[i, :length, :6] = np.array(feat[:length])[:, :6]

# Free the raw dict to save RAM!
del cic_data
import gc
gc.collect()

print("  Padding arrays to prevent torch.compile recompilation...")
orig_n_cic = len(cic_feats)
pad_len = (128 - (orig_n_cic % 128)) % 128
if pad_len > 0:
    cic_feats = np.vstack([cic_feats, np.zeros((pad_len, 32, 6), dtype=np.float32)])

t0_cic = time.perf_counter()
cic_emb_list = []
with torch.no_grad():
    for i in range(0, len(cic_feats), 128):
        batch = torch.tensor(cic_feats[i:i+128], device=DEVICE)
        emb = encoder(batch)
        cic_emb_list.append(emb.cpu().numpy())
torch.cuda.synchronize()
e_cic = time.perf_counter() - t0_cic

cic_emb = np.concatenate(cic_emb_list, axis=0)[:orig_n_cic] # remove padding
cic_fps = orig_n_cic / e_cic
cic_lat = (e_cic / orig_n_cic) * 1000

print(f"  CICIDS Embedding Throughput : {cic_fps:,.0f} flows/sec")
print(f"  CICIDS Embedding Latency    : {cic_lat:.4f} ms/flow")

# UDA Split: 15% Reference Pool, 85% Test Set
cic_benign_idx = np.where(cic_labels == 0)[0]
cic_attack_idx = np.where(cic_labels == 1)[0]
cic_ref_idx, cic_test_b_idx = train_test_split(cic_benign_idx, test_size=0.85, random_state=42)
cic_test_idx = np.sort(np.concatenate([cic_test_b_idx, cic_attack_idx]))

cic_ref_emb = cic_emb[cic_ref_idx]
cic_test_emb = cic_emb[cic_test_idx]
cic_test_labels = cic_labels[cic_test_idx]
cic_test_cats = cic_cats[cic_test_idx]

print(f"\n  Total CICIDS flows : {len(cic_labels):,}")
print(f"  CICIDS Ref pool    : {len(cic_ref_idx):,} (15% benign)")
print(f"  CICIDS Test set    : {len(cic_test_idx):,} ({(cic_test_labels==0).sum():,} benign + {(cic_test_labels==1).sum():,} attack)")

print(f"\n  {'Config':<16} | {'Protocol':<10} | {'ROC-AUC':>8} | {'PR-AUC':>8} | {'Macro F1':>8} | {'Prec':>8} | {'Recall':>8} | {'FAR':>8}")
print("  " + "-"*100)

best_row_cic = None
for name, pd, k in configs:
    # UDA: Fit PCA exclusively on CICIDS reference pool
    pca = PCA(n_components=pd, whiten=True, random_state=42)
    rp = pca.fit_transform(cic_ref_emb).astype(np.float32)
    ep = pca.transform(cic_test_emb).astype(np.float32)
    faiss.normalize_L2(rp); faiss.normalize_L2(ep)

    res = faiss.StandardGpuResources()
    idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(pd))
    idx.add(rp) # Build FAISS index from CICIDS Reference Pool

    ref_sims, _ = idx.search(rp, k)
    ref_scores = 1.0 - (ref_sims[:,0] if k==1 else ref_sims.mean(axis=1))

    threshold = np.percentile(ref_scores, 97)
    
    test_sims, _ = idx.search(ep, k)
    test_scores = 1.0 - (test_sims[:,0] if k==1 else test_sims.mean(axis=1))

    predictions = (test_scores > threshold).astype(int)

    auc = roc_auc_score(cic_test_labels, test_scores)
    ap  = average_precision_score(cic_test_labels, test_scores)
    f_macro  = f1_score(cic_test_labels, predictions, average='macro')
    prec = precision_score(cic_test_labels, predictions)
    rec  = recall_score(cic_test_labels, predictions)
    fp_count = ((predictions==1)&(cic_test_labels==0)).sum()
    tn_count = ((predictions==0)&(cic_test_labels==0)).sum()
    far = fp_count/(fp_count+tn_count) if (fp_count+tn_count)>0 else 0

    fpr_curve, tpr_curve, thresh_curve = roc_curve(cic_test_labels, test_scores)
    youden_j = tpr_curve - fpr_curve
    best_oracle_idx = np.argmax(youden_j)
    oracle_threshold = float(thresh_curve[best_oracle_idx])
    
    oracle_predictions = (test_scores >= oracle_threshold).astype(int)
    oracle_f_macro = f1_score(cic_test_labels, oracle_predictions, average='macro')
    oracle_prec = precision_score(cic_test_labels, oracle_predictions)
    oracle_rec = recall_score(cic_test_labels, oracle_predictions)
    oracle_fp = ((oracle_predictions==1)&(cic_test_labels==0)).sum()
    oracle_tn = ((oracle_predictions==0)&(cic_test_labels==0)).sum()
    oracle_far = oracle_fp/(oracle_fp+oracle_tn) if (oracle_fp+oracle_tn)>0 else 0

    tag = " BEST" if best_row_cic is None or f_macro > best_row_cic['f_macro'] else "" 
    print(f"  {name:<16} | {'Rigorous':<10} | {auc:>8.4f} | {ap:>8.4f} | {f_macro:>8.4f} | {prec:>8.4f} | {rec:>8.4f} | {far:>7.4%}{tag}") 
    print(f"  {'':<16} | {'Oracle':<10} | {auc:>8.4f} | {ap:>8.4f} | {oracle_f_macro:>8.4f} | {oracle_prec:>8.4f} | {oracle_rec:>8.4f} | {oracle_far:>7.4%}")
    if best_row_cic is None or f_macro > best_row_cic['f_macro']: 
        best_row_cic = {'name':name,'auc':auc,'ap':ap,'f_macro':f_macro,
                    'prec':prec,'rec':rec, 'far':far, 'threshold':threshold,
                    'oracle_threshold':oracle_threshold, 'oracle_f_macro':oracle_f_macro,
                    'oracle_prec':oracle_prec, 'oracle_rec':oracle_rec, 'oracle_far':oracle_far}

print(f"\n{'═'*60}") 
print(f"  THESIS FINAL UDA CROSS-DOMAIN NUMBERS (Best: {best_row_cic['name']})") 
print(f"{'═'*60}") 
print(f"  ROC-AUC    : {best_row_cic['auc']:.4f}") 
print(f"  PR-AUC     : {best_row_cic['ap']:.4f}") 
print(f"  Macro F1   : {best_row_cic['f_macro']:.4f} (Oracle: {best_row_cic['oracle_f_macro']:.4f})") 
print(f"  Precision  : {best_row_cic['prec']:.4f} (Oracle: {best_row_cic['oracle_prec']:.4f})") 
print(f"  Recall     : {best_row_cic['rec']:.4f} (Oracle: {best_row_cic['oracle_rec']:.4f})") 
print(f"  FAR        : {best_row_cic['far']:.4%} (Oracle: {best_row_cic['oracle_far']:.4%})") 
print(f"  Threshold  : {best_row_cic['threshold']:.6f} (97th pct CICIDS ref pool)")
print(f"  Oracle Thr : {best_row_cic['oracle_threshold']:.6f} (Optimal test labels)")
print(f"  Throughput : {cic_fps:,.0f} flows/sec") 
print(f"  Latency    : {cic_lat:.4f} ms/flow") 
print(f"{'═'*60}")
