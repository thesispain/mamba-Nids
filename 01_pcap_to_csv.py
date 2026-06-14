#!/usr/bin/env python3
"""
STAGE 1: PCAP → Flat Labeled CSV
=================================
Uses dpkt (pure Python) — NO tshark required.
Reads all 80 PCAPs from /home/T2510596/Downloads/DATA/
Outputs a single flat CSV: unsw_packets.csv

NOTE: UNSW-NB15 PCAPs use DLT_LINUX_SLL (link type 113), NOT standard
Ethernet (DLT_EN10MB=1). The SLL header is 16 bytes before the IP payload.

Labeling Strategy:
  - Load GT file. Build a list of attack time-windows per attacker IP.
  - A packet is "attack" if its src OR dst IP is one of the 4 known
    attacker IPs AND its timestamp falls within any GT window for that IP.
  - This avoids the catastrophic port/protocol matching failures of v1.
  - Everything else defaults to Benign.

Run ONCE. Never run again after unsw_packets.csv exists.
"""

import os
import csv
import socket
import struct
import dpkt
import time
from pathlib import Path
from collections import defaultdict

# Link layer type constants
DLT_EN10MB    = 1    # Standard Ethernet
DLT_LINUX_SLL = 113  # Linux cooked capture (used by UNSW-NB15 PCAPs)
DLT_RAW_1     = 12   # Raw IP v1
DLT_RAW_2     = 101  # Raw IP v2

# ── Config ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

PCAP_DIRS = [
    os.path.join(DATA_DIR, 'pcap-22.1.2015'),
    os.path.join(DATA_DIR, 'PCAP-17.2.2015'),
]
GT_FILE   = os.path.join(DATA_DIR, 'NUSW-NB15_GT.csv')
OUT_CSV   = os.path.join(DATA_DIR, 'unsw_packets.csv')
LOG_FILE  = os.path.join(DATA_DIR, 'stage1.log')

# Known UNSW-NB15 attacker IPs (fixed lab topology)
ATTACKER_IPS = {'175.45.176.0', '175.45.176.1', '175.45.176.2', '175.45.176.3'}
# Known victim IPs
VICTIM_IPS   = {f'149.171.126.{i}' for i in range(10, 20)}

# Fuzzy time tolerance: GT windows can be slightly off from PCAP timestamps
GT_SLACK_SEC = 2.0
# ─────────────────────────────────────────────────────────────────────────

def log(msg, file=None):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    if file:
        file.write(line + '\n')
        file.flush()

def ip_to_str(raw_bytes):
    """Convert 4-byte raw IP to dotted string."""
    try:
        return socket.inet_ntoa(raw_bytes)
    except:
        return '0.0.0.0'

def load_gt(gt_path):
    """
    Load GT file into a fast lookup structure.
    Returns: dict {src_ip → [(start_ts, end_ts, category), ...]}
    Indexed by ATTACKER IP only. We check src and dst at runtime.
    """
    windows = defaultdict(list)
    skipped = 0
    loaded = 0

    with open(gt_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 9:
                continue
            try:
                start_t  = float(row[0].strip())
                end_t    = float(row[1].strip())
                category = row[2].strip()
                src_ip   = row[5].strip()
                dst_ip   = row[7].strip()

                # Add slack to time window
                start_t -= GT_SLACK_SEC
                end_t   += GT_SLACK_SEC

                # Index by the attacker IP (src in GT is always attacker)
                if src_ip in ATTACKER_IPS:
                    windows[src_ip].append((start_t, end_t, category))
                    loaded += 1
                else:
                    skipped += 1
            except (ValueError, IndexError):
                skipped += 1

    print(f'  GT loaded: {loaded:,} windows, {skipped:,} skipped')
    for ip, wins in windows.items():
        print(f'    {ip}: {len(wins):,} attack windows')
    return windows

def is_attack(src_ip, dst_ip, timestamp, gt_windows):
    """
    Fast IP-based labeling. Returns (is_attack: bool, category: str).
    Check if either the src or dst IP is an attacker IP with an
    active GT time window at this timestamp.
    """
    # Check src as attacker
    if src_ip in gt_windows:
        for (t_start, t_end, cat) in gt_windows[src_ip]:
            if t_start <= timestamp <= t_end:
                return True, cat

    # Check dst as attacker (rare — response packet)
    if dst_ip in gt_windows:
        for (t_start, t_end, cat) in gt_windows[dst_ip]:
            if t_start <= timestamp <= t_end:
                return True, cat

    return False, 'Benign'

def parse_ip_from_buf(buf, datalink):
    """
    Parse raw PCAP buffer into a dpkt.ip.IP object.
    Handles multiple link layer types:
      1   = DLT_EN10MB  (standard Ethernet, 14-byte header)
      113 = DLT_LINUX_SLL (Linux cooked, 16-byte header)
      12/101 = DLT_RAW (raw IP, no header)
    Returns None for non-IPv4 or on error.
    """
    try:
        if datalink == DLT_EN10MB:
            eth = dpkt.ethernet.Ethernet(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                return None
            return eth.data
        elif datalink == DLT_LINUX_SLL:
            # SLL header: 2+2+2+8+2 = 16 bytes total
            # EtherType is at bytes 14-15
            ethertype = struct.unpack('!H', buf[14:16])[0]
            if ethertype == 0x0800:  # IPv4
                return dpkt.ip.IP(buf[16:])
            return None  # skip IPv6, ARP, etc.
        elif datalink in (DLT_RAW_1, DLT_RAW_2):
            return dpkt.ip.IP(buf)
        return None
    except:
        return None

def get_ports_and_flags(ip):
    """Extract src_port, dst_port, tcp_flags from a dpkt.ip.IP object."""
    src_port = 0
    dst_port = 0
    flags    = 0
    try:
        transport = ip.data
        if isinstance(transport, dpkt.tcp.TCP):
            src_port = transport.sport
            dst_port = transport.dport
            flags    = transport.flags
        elif isinstance(transport, dpkt.udp.UDP):
            src_port = transport.sport
            dst_port = transport.dport
    except:
        pass
    return src_port, dst_port, flags

def process_pcap(pcap_path, gt_windows, csv_writer, log_f):
    """Process a single PCAP file. Returns (total, attack, non_ip, errors)."""
    total      = 0
    attack_pkt = 0
    non_ip     = 0
    errors     = 0

    try:
        with open(pcap_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            datalink = pcap.datalink()  # detect link layer once per file

            for ts, buf in pcap:
                total += 1
                try:
                    ip = parse_ip_from_buf(buf, datalink)
                    if ip is None:
                        non_ip += 1
                        continue

                    src_ip    = ip_to_str(ip.src)
                    dst_ip    = ip_to_str(ip.dst)
                    proto     = ip.p
                    frame_len = len(buf)

                    src_port, dst_port, tcp_flags = get_ports_and_flags(ip)

                    # Label the packet using IP+timestamp fuzzy matching
                    pkt_is_attack, category = is_attack(src_ip, dst_ip, ts, gt_windows)
                    if pkt_is_attack:
                        attack_pkt += 1

                    csv_writer.writerow([
                        f'{ts:.6f}',
                        src_ip,
                        dst_ip,
                        src_port,
                        dst_port,
                        proto,
                        frame_len,
                        tcp_flags,
                        1 if pkt_is_attack else 0,
                        category
                    ])

                except Exception:
                    errors += 1
                    continue

    except Exception as e:
        log(f'  ERROR reading {pcap_path}: {e}', log_f)

    return total, attack_pkt, non_ip, errors

# ── MAIN ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    if os.path.exists(OUT_CSV):
        print(f'✅ Output already exists: {OUT_CSV}')
        print('   Delete it to re-run Stage 1.')
        exit(0)

    log_f = open(LOG_FILE, 'w')
    t_start_all = time.time()

    log('=== STAGE 1: PCAP → CSV ===', log_f)

    # 1. Load GT
    log('Loading Ground Truth...', log_f)
    gt_windows = load_gt(GT_FILE)

    # 2. Find all PCAPs
    all_pcaps = []
    for d in PCAP_DIRS:
        all_pcaps.extend(sorted(Path(d).glob('*.pcap')))
    log(f'Found {len(all_pcaps)} PCAP files', log_f)

    # 3. Open output CSV
    with open(OUT_CSV, 'w', newline='') as out_f:
        writer = csv.writer(out_f)
        # Write header
        writer.writerow([
            'timestamp', 'src_ip', 'dst_ip',
            'src_port', 'dst_port', 'proto',
            'frame_len', 'tcp_flags',
            'label', 'attack_cat'
        ])

        total_pkts   = 0
        total_attack = 0
        total_non_ip = 0
        total_errors = 0

        for i, pcap_path in enumerate(all_pcaps):
            t0 = time.time()
            n, a, ni, e = process_pcap(str(pcap_path), gt_windows, writer, log_f)
            elapsed = time.time() - t0
            total_pkts   += n
            total_attack += a
            total_non_ip += ni
            total_errors += e
            log(f'  [{i+1:02d}/{len(all_pcaps)}] {pcap_path.name}: '
                f'{n:>8,} pkts  {a:>7,} attack  {elapsed:.0f}s', log_f)

    elapsed_total = time.time() - t_start_all
    log('', log_f)
    log('=== STAGE 1 COMPLETE ===', log_f)
    log(f'  Total packets:    {total_pkts:>12,}', log_f)
    log(f'  Attack packets:   {total_attack:>12,}  ({100*total_attack/max(1,total_pkts):.2f}%)', log_f)
    log(f'  Non-IP skipped:   {total_non_ip:>12,}', log_f)
    log(f'  Parse errors:     {total_errors:>12,}', log_f)
    log(f'  Output:           {OUT_CSV}', log_f)
    log(f'  Time:             {elapsed_total/60:.1f} minutes', log_f)
    log_f.close()
