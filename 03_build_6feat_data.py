#!/usr/bin/env python3
"""
50_build_champion_data.py — Pure Absolute Log Dataset
======================================================
No delta. No hybrid. Pure absolute magnitudes for ALL 32 packets.
Train: 100% Benign. Eval: Everything (all attacks = zero-day).
"""
import os, pickle, time
import numpy as np
from collections import defaultdict, Counter

ABSLOG_DIR = '/home/T2510596/1TB_Storage_new/thesis-mamba-nids/data/data_abslog'
OUT_DIR    = '/home/T2510596/1TB_Storage_new/thesis-mamba-nids/data/data_champion'
CSV_PATH   = '/home/T2510596/1TB_Storage_new/DATA/unsw_packets.csv'
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42; SEQ_LEN = 32

def port_category(port):
    port = int(port)
    if port < 1024: return 0
    elif port < 49152: return 1
    else: return 2

def build_absolute_features(packets):
    packets = sorted(packets, key=lambda p: p['ts'])[:SEQ_LEN]
    n = len(packets)
    features = np.zeros((SEQ_LEN, 6), dtype=np.float32)
    for i, pkt in enumerate(packets):
        features[i, 0] = float(pkt['proto'])
        features[i, 1] = np.log1p(float(pkt['frame_len']))
        features[i, 2] = float(pkt['flags'])
        if i == 0:
            features[i, 3] = 0.0
        else:
            iat = max(pkt['ts'] - packets[i-1]['ts'], 1e-6)
            features[i, 3] = np.log(iat + 1e-6)
        features[i, 4] = float(pkt['direction'])
        features[i, 5] = float(pkt['port_cat'])
    return features, n

print("="*70)
print("  CHAMPION DATA: Pure Absolute Log — No Delta, No Hybrid")
print("="*70)

# Check if abslog data already exists from the previous pipeline
abslog_train = os.path.join(ABSLOG_DIR, 'unsw_abslog_train.pkl')
abslog_eval  = os.path.join(ABSLOG_DIR, 'unsw_abslog_eval.pkl')

if os.path.exists(abslog_train) and os.path.exists(abslog_eval):
    print("\n  Found existing absolute-log data. Re-partitioning...")
    print(f"  Loading {abslog_train}...")
    with open(abslog_train, 'rb') as f:
        part1 = pickle.load(f)
    print(f"  Loading {abslog_eval}...")
    with open(abslog_eval, 'rb') as f:
        part2 = pickle.load(f)

    all_data = part1 + part2
    del part1, part2
    print(f"  Combined: {len(all_data):,} flows")

else:
    print(f"\n  No cached data. Rebuilding from CSV: {CSV_PATH}")
    import pandas as pd
    flows = defaultdict(lambda: {'packets': [], 'labels': [], 'cats': []})
    total_packets = 0
    t0 = time.time()

    for chunk in pd.read_csv(CSV_PATH, chunksize=2_000_000, low_memory=False):
        total_packets += len(chunk)
        if total_packets % 10_000_000 == 0:
            print(f"  Processed {total_packets:,} packets...", flush=True)
        for _, row in chunk.iterrows():
            sip, dip = str(row['src_ip']), str(row['dst_ip'])
            sp, dp = int(row['src_port']), int(row['dst_port'])
            proto = int(row['proto'])
            key = (min(sip,dip), max(sip,dip), min(sp,dp), max(sp,dp), proto)
            direction = 0 if sip <= dip else 1
            flows[key]['packets'].append({
                'ts': float(row['timestamp']), 'frame_len': int(row['frame_len']),
                'proto': proto, 'flags': int(row.get('tcp_flags', 0)),
                'direction': direction, 'port_cat': port_category(dp)
            })
            flows[key]['labels'].append(int(row['label']))
            cat = str(row.get('attack_cat', 'Normal')).strip()
            if cat in ('', 'nan', 'Normal'): cat = 'Benign'
            flows[key]['cats'].append(cat)

    print(f"  Packets: {total_packets:,} | Flows: {len(flows):,} | Time: {time.time()-t0:.0f}s")
    print("  Building features...")
    all_data = []
    for key, fdata in flows.items():
        features, n_real = build_absolute_features(fdata['packets'])
        label = 1 if sum(fdata['labels']) > len(fdata['labels']) / 2 else 0
        cat = Counter(fdata['cats']).most_common(1)[0][0]
        if label == 0: cat = 'Benign'
        all_data.append({'features': features, 'label': label, 'attack_cat': cat, 'n_real_packets': n_real})
    del flows

# Re-partition: Train = 100% Benign, Eval = ALL
np.random.seed(SEED)
benign = [d for d in all_data if d['label'] == 0]
attacks = [d for d in all_data if d['label'] == 1]
del all_data

np.random.shuffle(benign)
split = len(benign) // 2
train_data = benign[:split]
eval_data = benign[split:] + attacks
np.random.shuffle(eval_data)

print(f"\n  TRAIN: {len(train_data):,} flows (100% Benign)")
print(f"  EVAL:  {len(eval_data):,} flows")
eval_cats = Counter(d['attack_cat'] for d in eval_data)
for cat, n in sorted(eval_cats.items(), key=lambda x: -x[1]):
    print(f"    {cat:20s}: {n:>8,}")

# Verify encoding
s = train_data[0]['features']
n_real = int(np.sum(~np.all(s == 0, axis=1)))
print(f"\n  Encoding check (first train flow, {n_real} packets):")
for i in range(min(4, n_real)):
    print(f"    Pkt {i}: LogLen={s[i,1]:.4f} LogIAT={s[i,3]:.4f}  ← ALL ABSOLUTE")

# Save
train_path = os.path.join(OUT_DIR, 'champion_train.pkl')
eval_path  = os.path.join(OUT_DIR, 'champion_eval.pkl')
with open(train_path, 'wb') as f:
    pickle.dump(train_data, f, protocol=4)
with open(eval_path, 'wb') as f:
    pickle.dump(eval_data, f, protocol=4)

print(f"\n  Saved: {train_path} ({os.path.getsize(train_path)/1e6:.0f} MB)")
print(f"  Saved: {eval_path} ({os.path.getsize(eval_path)/1e6:.0f} MB)")
print("\n✅ Champion data built.")
