#!/usr/bin/env python3
import os
import csv
import pickle
import numpy as np
import math
import time
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

IN_CSV          = os.path.join(DATA_DIR, 'unsw_packets.csv')
OUT_DIR         = Path(os.path.join(DATA_DIR, 'unswnb15_full'))
OUT_PRETRAIN    = OUT_DIR / 'pretrain_benign.pkl'
OUT_EVAL        = OUT_DIR / 'eval_mixed.pkl'
LOG_FILE        = os.path.join(DATA_DIR, 'stage2.log')

MAX_PACKETS     = 32
CHUNK_SIZE      = 500_000
RANDOM_SEED     = 42

def log(msg, f=None):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    if f:
        f.write(line + '\n')
        f.flush()

def build_flow_matrix(packets, max_packets=32):
    n = len(packets)

    times   = np.array([p[0] for p in packets], dtype=np.float64)
    lens    = np.array([p[4] for p in packets], dtype=np.float64)
    flags   = np.array([p[5] for p in packets], dtype=np.float64)
    proto   = np.full(n, packets[0][3], dtype=np.float64)
    is_first_src = np.array([p[6] for p in packets], dtype=np.float64)

    log_len     = np.log1p(lens)
    delta_loglen = np.diff(log_len, prepend=log_len[0])

    iat         = np.diff(times, prepend=times[0])
    iat         = np.clip(iat, 0, None)
    log_iat     = np.log1p(iat)
    delta_logiat = np.diff(log_iat, prepend=log_iat[0])

    first_src_flag = is_first_src[0]
    direction = (is_first_src != first_src_flag).astype(np.float64)

    matrix = np.column_stack([proto, delta_loglen, flags, delta_logiat, direction])

    if n < max_packets:
        pad = np.zeros((max_packets - n, 5), dtype=np.float32)
        matrix = np.vstack([matrix, pad])
    else:
        matrix = matrix[:max_packets]

    return matrix.astype(np.float32)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_f = open(LOG_FILE, 'w')
    t_all = time.time()

    log('=== STAGE 2: CSV to Flow PKLs ===', log_f)
    log(f'  Input:   {IN_CSV}', log_f)
    log(f'  OutDir:  {OUT_DIR}', log_f)

    log('Pass 1: Grouping packets into flows...', log_f)

    flow_map    = defaultdict(list)
    flow_labels = {}
    flow_cats   = {}

    pkt_total   = 0
    pkt_attack  = 0
    chunk_num   = 0

    with open(IN_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader)

        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) >= CHUNK_SIZE:
                chunk_num += 1
                for r in batch:
                    try:
                        ts        = float(r[0])
                        src_ip    = r[1]
                        dst_ip    = r[2]
                        src_port  = int(r[3])
                        dst_port  = int(r[4])
                        proto     = int(r[5])
                        frame_len = int(r[6])
                        tcp_flags = int(r[7])
                        label     = int(r[8])
                        cat       = r[9]

                        if src_ip <= dst_ip:
                            key = (src_ip, dst_ip, src_port, dst_port, proto)
                            is_first_src = True
                        else:
                            key = (dst_ip, src_ip, dst_port, src_port, proto)
                            is_first_src = False

                        flow_map[key].append((ts, src_ip, src_port, proto, frame_len, tcp_flags, is_first_src))

                        if label == 1:
                            flow_labels[key] = 1
                            flow_cats[key]   = cat
                        elif key not in flow_labels:
                            flow_labels[key] = 0
                            flow_cats[key]   = 'Benign'

                        pkt_total  += 1
                        pkt_attack += label
                    except (ValueError, IndexError):
                        continue
                batch = []
                if chunk_num % 10 == 0:
                    log(f'  Chunk {chunk_num}: {pkt_total:,} pkts, {len(flow_map):,} flows so far', log_f)

        for r in batch:
            try:
                ts        = float(r[0])
                src_ip    = r[1]
                dst_ip    = r[2]
                src_port  = int(r[3])
                dst_port  = int(r[4])
                proto     = int(r[5])
                frame_len = int(r[6])
                tcp_flags = int(r[7])
                label     = int(r[8])
                cat       = r[9]

                if src_ip <= dst_ip:
                    key = (src_ip, dst_ip, src_port, dst_port, proto)
                    is_first_src = True
                else:
                    key = (dst_ip, src_ip, dst_port, src_port, proto)
                    is_first_src = False

                flow_map[key].append((ts, src_ip, src_port, proto, frame_len, tcp_flags, is_first_src))
                if label == 1:
                    flow_labels[key] = 1
                    flow_cats[key]   = cat
                elif key not in flow_labels:
                    flow_labels[key] = 0
                    flow_cats[key]   = 'Benign'
                pkt_total  += 1
                pkt_attack += label
            except (ValueError, IndexError):
                continue

    log(f'Pass 1 complete:', log_f)
    log(f'  Total packets:  {pkt_total:,}', log_f)
    log(f'  Attack packets: {pkt_attack:,}  ({100*pkt_attack/max(1,pkt_total):.2f}%)', log_f)
    log(f'  Total flows:    {len(flow_map):,}', log_f)

    log('Pass 2: Building flow matrices...', log_f)

    benign_flows = []
    attack_flows = []

    for i, (key, packets) in enumerate(flow_map.items()):
        if i % 100_000 == 0:
            log(f'  Processing flow {i:,} / {len(flow_map):,}', log_f)

        packets.sort(key=lambda p: p[0])

        matrix = build_flow_matrix(packets, MAX_PACKETS)
        lbl    = flow_labels[key]
        cat    = flow_cats[key]

        flow_obj = {
            'features':    matrix,
            'label':       lbl,
            'attack_cat':  cat,
        }

        if lbl == 0:
            benign_flows.append(flow_obj)
        else:
            attack_flows.append(flow_obj)

    del flow_map

    log(f'Pass 2 complete:', log_f)
    log(f'  Benign flows: {len(benign_flows):,}', log_f)
    log(f'  Attack flows: {len(attack_flows):,}', log_f)

    log('Applying strict 50/50 benign split...', log_f)

    random.seed(RANDOM_SEED)
    random.shuffle(benign_flows)

    split_idx       = len(benign_flows) // 2
    pretrain_flows  = benign_flows[:split_idx]
    eval_benign     = benign_flows[split_idx:]
    eval_flows      = eval_benign + attack_flows

    log(f'  pretrain_benign:  {len(pretrain_flows):,} flows  (100% benign)', log_f)
    log(f'  eval_mixed:       {len(eval_flows):,} flows', log_f)
    log(f'    Benign:         {len(eval_benign):,}', log_f)
    log(f'    Attack:         {len(attack_flows):,}', log_f)

    log(f'Saving pretrain_benign.pkl...', log_f)
    with open(OUT_PRETRAIN, 'wb') as f:
        pickle.dump(pretrain_flows, f, protocol=pickle.HIGHEST_PROTOCOL)

    log(f'Saving eval_mixed.pkl...', log_f)
    random.shuffle(eval_flows)
    with open(OUT_EVAL, 'wb') as f:
        pickle.dump(eval_flows, f, protocol=pickle.HIGHEST_PROTOCOL)

    elapsed = (time.time() - t_all) / 60
    log('', log_f)
    log('=== STAGE 2 COMPLETE ===', log_f)
    log(f'  pretrain_benign.pkl -> {OUT_PRETRAIN}', log_f)
    log(f'  eval_mixed.pkl      -> {OUT_EVAL}', log_f)
    log(f'  Total time: {elapsed:.1f} minutes', log_f)
    log_f.close()

if __name__ == '__main__':
    main()
