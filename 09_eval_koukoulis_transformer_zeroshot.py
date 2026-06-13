import torch, pickle, numpy as np, faiss, gc
import torch.nn as nn, torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, f1_score, recall_score, roc_curve

DEVICE = torch.device('cuda')
DATA_DIR = '/media/T2510596/HDD/thesis-mamba-nids/data/data_6feat'
CKPT = '/media/T2510596/HDD/thesis-mamba-nids/checkpoints/koukoulis_exact_4layer.pt'

# Exact Koukoulis architecture from their paper
import math

class EmbeddingLayer(nn.Module):
    def __init__(self, hidden_dim=256, seq_len=32, num_features=5, 
                 num_columns=[0,3], dict_size=65536):
        super().__init__()
        self.num_features = num_features
        self.num_columns = num_columns
        self.seq_len = seq_len
        self.cat_mask = torch.ones(num_features, dtype=torch.bool)
        self.cat_mask[num_columns] = False
        self.num_mask = ~self.cat_mask
        self.norm = nn.LayerNorm(hidden_dim)
        self.cls_token = nn.Parameter(torch.empty((1,1,hidden_dim)))
        proj_dim = hidden_dim // num_features
        self.position_embeddings = nn.Embedding(seq_len+1, hidden_dim)
        n_cat = num_features - len(num_columns)
        self.cat_emb_layer = nn.ModuleList([
            nn.Embedding(dict_size, proj_dim) for _ in range(n_cat)])
        self.num_emb_layer = nn.ModuleList([
            nn.Linear(1, proj_dim, bias=False) for _ in range(len(num_columns))])
        self.proj_layer = nn.Linear(proj_dim * num_features, hidden_dim, bias=False)
    def forward(self, x):
        num_input = x[:,:,self.num_mask]
        num_emb = torch.cat([self.num_emb_layer[i](num_input[:,:,[i]]) 
                             for i in range(num_input.shape[-1])], dim=2)
        cat_input = x[:,:,self.cat_mask].long()
        cat_emb = torch.cat([self.cat_emb_layer[i](cat_input[:,:,i]) 
                             for i in range(cat_input.shape[-1])], dim=2)
        embed_tokens = self.proj_layer(torch.cat((num_emb, cat_emb), dim=2))
        B = x.shape[0]
        cls = self.cls_token.expand(B,-1,-1)
        embed_tokens = torch.cat((cls, embed_tokens), dim=1)
        pos = torch.arange(embed_tokens.shape[1], device=x.device).unsqueeze(0).expand(B,-1)
        return self.norm(embed_tokens + self.position_embeddings(pos))

class BERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = EmbeddingLayer()
        self.encoder_layer = nn.TransformerEncoderLayer(
            256, 4, 1024, batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=4)
    def embeddings(self, x, mask=None):
        embed = self.embed(x)
        B = x.shape[0]
        if mask is not None:
            cls_mask = torch.zeros((B,1), dtype=torch.bool, device=x.device)
            transformer_mask = torch.cat((cls_mask, mask), dim=1)
            enc = self.encoder(embed, src_key_padding_mask=transformer_mask)
        else:
            enc = self.encoder(embed)
        return enc[:,0,:]

# Load model
model = BERT().to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print("Koukoulis Transformer loaded")

def get_emb_koukoulis(data, max_flows=None):
    if max_flows: data=data[:max_flows]
    # Convert from 6-feat log to Koukoulis 5-feat format
    # Their features: [timestamp(IAT), size, ip_protocol, direction, tcp_flags]
    # Our features: [proto, log_len, flags, log_iat, direction, port_cat]
    # Map: col0=proto->col2, col1=log_len->size, col2=flags->col4, col3=log_iat->col0, col4=dir->col3
    embs = []
    chunk = 1024
    with torch.no_grad():
        for i in range(0, len(data), chunk):
            batch = data[i:i+chunk]
            feats = np.array([d['features'] if isinstance(d,dict) else d 
                             for d in batch], dtype=np.float32)
            # Reorder to Koukoulis format: [iat, len, proto, dir, flags]
            mapped = np.zeros((len(feats), 32, 5), dtype=np.float32)
            mapped[:,:,0] = feats[:,:,3]  # iat
            mapped[:,:,1] = feats[:,:,1]  # len
            mapped[:,:,2] = feats[:,:,0]  # proto
            mapped[:,:,3] = feats[:,:,4]  # direction
            mapped[:,:,4] = feats[:,:,2]  # flags
            t = torch.from_numpy(mapped).to(DEVICE)
            embs.append(model.embeddings(t).cpu().numpy())
            del feats, mapped, t; gc.collect()
    return np.concatenate(embs)

def score(train_emb, test_emb, test_labels, name):
    pca = PCA(n_components=12, whiten=True, random_state=42)
    tp = pca.fit_transform(train_emb).astype(np.float32)
    ep = pca.transform(test_emb).astype(np.float32)
    faiss.normalize_L2(tp); faiss.normalize_L2(ep)
    res = faiss.StandardGpuResources()
    idx = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(12))
    idx.add(tp)
    sims,_ = idx.search(ep, 1)
    scores = 1.0 - sims[:,0]
    auc = roc_auc_score(test_labels, scores)
    fpr,tpr,th = roc_curve(test_labels, scores)
    bi = np.argmax(tpr-fpr)
    preds = (scores>th[bi]).astype(int)
    mf1 = f1_score(test_labels, preds, average='macro', zero_division=0)
    rec = recall_score(test_labels, preds, zero_division=0)
    fp = ((preds==1)&(test_labels==0)).sum()
    tn = ((preds==0)&(test_labels==0)).sum()
    far = fp/(fp+tn)*100
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  AUC:{auc:.4f} F1:{mf1:.4f} Rec:{rec:.4f} FAR:{far:.2f}%")
    return auc

# Load UNSW train for reference pool
print("Loading UNSW train...")
with open(f'{DATA_DIR}/unsw_hybrid_train.pkl','rb') as f: train=pickle.load(f)
train_emb = get_emb_koukoulis(train[:100000])
del train; gc.collect()

# UNSW in-domain
print("UNSW eval...")
with open(f'{DATA_DIR}/unsw_hybrid_eval.pkl','rb') as f: evl=pickle.load(f)
unsw_labels = np.array([d['label'] for d in evl])
unsw_emb = get_emb_koukoulis(evl)
del evl; gc.collect()
score(train_emb, unsw_emb, unsw_labels, "Koukoulis Transformer UNSW->UNSW")
del unsw_emb, unsw_labels; gc.collect()

# CICIDS cross-domain
print("CICIDS eval...")
with open(f'{DATA_DIR}/cicids_6feat_eval_v2.pkl','rb') as f: cic=pickle.load(f)
cic_labels = np.array([d.get('label',0) if isinstance(d,dict) else 0 for d in cic])
cic_emb = get_emb_koukoulis(cic)
del cic; gc.collect()
score(train_emb, cic_emb, cic_labels, "Koukoulis Transformer UNSW->CICIDS")

print("\nDONE")
