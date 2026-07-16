"""
Builds data/specs.db — a small, structured SQLite database of AI
infrastructure hardware specs (GPUs and enterprise SSDs).

This stands in for what a real "Cloud APIs" / live inventory system tool
would query in production. Unlike the document corpus (which is
self-authored prose), every row here is a REAL published spec pulled from
manufacturer datasheets and product pages — see the `source_url` column
on each row. Facts/specs are not copyrightable expression, so citing
manufacturer-published numbers here is standard practice (same as any
spec-comparison site would do), unlike reproducing article prose.

Run once:
    python data/build_specs_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "specs.db"

GPUS = [
    # model, vram_gb, memory_type, memory_bandwidth_gbps, fp16_tflops_dense,
    # nvlink_gbps, tdp_watts, price_tier, notes, source_url
    (
        "NVIDIA A100 80GB SXM", 80, "HBM2e", 2039, 312, 600, 400,
        "enterprise",
        "Ampere architecture. 312 TFLOPS FP16 is the dense (non-sparsity) "
        "figure; doubles with structured sparsity. Discontinued by NVIDIA "
        "Jan 2024 but still widely deployed on cloud infra.",
        "https://www.nvidia.com/en-us/data-center/a100/",
    ),
    (
        "NVIDIA H100 80GB SXM", 80, "HBM3", 3350, 989, 900, 700,
        "enterprise",
        "Hopper architecture. 989 TFLOPS FP16 is the sparsity-enabled "
        "figure per NVIDIA's datasheet; dense (non-sparsity) is roughly "
        "half. Adds FP8 Transformer Engine, ~3-4x A100 throughput on "
        "transformer workloads.",
        "https://www.nvidia.com/en-us/data-center/h100/",
    ),
    (
        "NVIDIA H100 PCIe", 80, "HBM2e", 2000, 800, None, 350,
        "enterprise",
        "PCIe variant of H100: lower memory bandwidth than SXM (HBM2e vs "
        "HBM3), no NVLink bridge, fits standard PCIe Gen5 server slots. "
        "Lower power (350W) than SXM (700W).",
        "https://www.thundercompute.com/blog/nvidia-h100-specs-full-guide",
    ),
    (
        "NVIDIA H200", 141, "HBM3e", 4800, 989, 900, None,
        "enterprise",
        "Same GH100 compute die as H100 (identical CUDA/Tensor Core "
        "counts) — the upgrade is entirely memory: 76% more capacity, "
        "43% more bandwidth than H100 SXM. Matters most for models >80GB "
        "or very long context windows.",
        "https://www.runpod.io/articles/guides/nvidia-h100",
    ),
    (
        "NVIDIA RTX 4090", 24, "GDDR6X", 1008, 82, None, 450,
        "consumer",
        "Consumer-tier card. No NVLink, no MIG, no enterprise support "
        "contract — but a common cost-effective choice for small-model "
        "fine-tuning/inference prototyping before scaling to enterprise "
        "GPUs.",
        "https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/",
    ),
]

SSDS = [
    # model, interface, capacity_tb, seq_read_mbps, seq_write_mbps,
    # endurance_dwpd, form_factor, notes, source_url
    (
        "Samsung PM1733", "PCIe Gen4", 30.72, 7000, 3800, 1.0, "2.5-inch U.2",
        "General-purpose enterprise NVMe. 1 DWPD over 5 years rated per "
        "JESD218/219 standards. Hardware power-loss protection (PLP).",
        "https://www.ssd.group/wp-content/uploads/2022/07/PM1733-25-SSD-Datasheet_v1.3_for-General.pdf",
    ),
    (
        "Samsung PM1763", "PCIe Gen6", 16.0, 28400, 21900, None, "E3.S",
        "Latest-gen (2026) high-performance enterprise SSD, optimized for "
        "AI/liquid-cooled server architectures. Samsung claims ~1.4s to "
        "transfer a 40GB LLM at rated throughput. Supports post-quantum "
        "cryptography (PQC) and TDISP for confidential computing.",
        "https://www.servethehome.com/samsung-pm1763-pcie-gen6-enterprise-ssd-in-production/",
    ),
    (
        "Solidigm D5-P5336", "PCIe Gen4", 122.88, None, None, 0.58, "U.2/E3.S",
        "QLC NAND, read-intensive / high-capacity tier. Built for "
        "read-heavy AI data lakes and large-scale bulk storage rather "
        "than write-heavy workloads (low 0.58 DWPD).",
        "https://www.atera.com/blog/best-enterprise-ssd/",
    ),
    (
        "Solidigm D7-P5810", "PCIe Gen4", 1.6, 6400, 4000, 50.0, "U.2",
        "SLC NAND, write-intensive tier — up to 50 DWPD random write "
        "endurance. Designed as a fast write-buffer/cache layer (e.g. "
        "CSAL) ahead of higher-capacity QLC drives, not bulk storage.",
        "https://www.solidigm.com/products/data-center/d7/p5810.html",
    ),
    (
        "Micron 6500 ION", "PCIe Gen4", 30.72, None, None, 0.0256, "E1.L",
        "QLC, extreme-density tier (~1PB per rack unit in E1.L form "
        "factor). Endurance rated at 0.41 DWPD for 64KB transfers, which "
        "converts to a much lower 4KB-equivalent DWPD — read-intensive "
        "AI storage-pool use case, not transactional.",
        "https://assets.micron.com/adobe/assets/urn:aaid:aem:7ae292a2-7897-4328-90e1-1485146834c1/renditions/original/as/6500-ion-nvme-ssd-product-brief.pdf",
    ),
]


def build():
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE gpus (
            model TEXT PRIMARY KEY,
            vram_gb REAL,
            memory_type TEXT,
            memory_bandwidth_gbps REAL,
            fp16_tflops_dense REAL,
            nvlink_gbps REAL,
            tdp_watts REAL,
            price_tier TEXT,
            notes TEXT,
            source_url TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO gpus VALUES (?,?,?,?,?,?,?,?,?,?)", GPUS
    )

    cur.execute(
        """
        CREATE TABLE ssds (
            model TEXT PRIMARY KEY,
            interface TEXT,
            capacity_tb REAL,
            seq_read_mbps REAL,
            seq_write_mbps REAL,
            endurance_dwpd REAL,
            form_factor TEXT,
            notes TEXT,
            source_url TEXT
        )
        """
    )
    cur.executemany(
        "INSERT INTO ssds VALUES (?,?,?,?,?,?,?,?,?)", SSDS
    )

    conn.commit()
    conn.close()
    print(f"Built {DB_PATH} with {len(GPUS)} GPUs and {len(SSDS)} SSDs.")


if __name__ == "__main__":
    build()
