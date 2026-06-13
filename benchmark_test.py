import time, torch
from torch import nn
from torch.nn import functional as F

D_MODEL = 256; N_LAYERS = 2; D_STATE = 16; EXPAND = 2; D_CONV = 4
DEVICE = 'cuda'

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
        self.layers = nn.ModuleList([PurePyTorchMamba(d_model, D_STATE, D_CONV, EXPAND) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        h = self.embedder(x)
        for layer in self.layers: h = layer(h) + h
        return self.norm(h)

encoder = MambaEncoder().to(DEVICE)
encoder.eval()
encoder = torch.compile(encoder, mode='reduce-overhead')

n = 100_000
dummy = torch.randn(n, 32, 6, device=DEVICE)

with torch.no_grad(), torch.amp.autocast('cuda'):
    for i in range(20): _ = encoder(dummy[i*128:(i+1)*128])
torch.cuda.synchronize()

for bs in [16, 32, 64, 128, 256, 512, 1024, 2048]:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for i in range(0, n, bs):
            _ = encoder(dummy[i:i+bs])
    torch.cuda.synchronize()
    e = time.perf_counter() - t0
    print(f"Batch {bs:>4}: {n/e:>10,.0f} flows/sec")
