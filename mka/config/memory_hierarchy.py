"""
Memory hierarchy and data paths for MKA / Block-MKA, this code is corresponds to the paper's (§4.2 Block-MKA, Fig. 2):
  • L1 — On-chip SRAM: tiled block attention, online softmax / m,z updates
    (FlashAttention-style scan); realized by SDPA/flash or custom CUDA kernels.
  • L2 — HBM: intermediate activations, Q/K/V, routed (fused) KV cache, softmax stats.
  • L3 — DRAM (vectorized hash, chunk recall): long-term retrieval / historical blocks;
    optional host or mmap-backed buffers when experiments wire retrieval.

FastMKA (Algorithm 2): route-fusion mixes L1/L2/(L3) before a single K,V projection,
then one causal attention — data flows as fused hidden states → KV in L2 → attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class MemoryHierarchyConfig:
    """Tier flags and paths for reproducible hierarchy-aware runs."""

    # --- Paper Block-MKA tiers → runtime ---
    l1_onchip_tiles: bool = True
    """L1: tiled / online attention in fast on-chip paths (kernel fusion, SDPA)."""

    l2_hbm_tensor_path: bool = True
    """L2: GPU HBM holds activations, fused KV, and attention working set."""

    l3_dram_chunk_recall: bool = False
    """L3: DRAM-tier chunk/hash recall (retrieval); enable when L3 retrieval is wired."""

    l3_recall_top_r: int = 0
    """Paper R: top-R chunks per query when L3 is used (0 = disabled / not configured)."""

    # --- Host / storage extensions (below HBM in the memory stack) ---
    host_dram_staging: bool = False
    """Pinned or pageable host RAM for staging before H2D or after D2H."""

    host_dram_pinned: bool = False

    ssd_spill_path: Optional[str] = None
    """Optional NVMe path for cold spill, mmap KV extensions, or ZeRO-style offload."""

    # --- Measurement (paper §6 inference) ---
    measure_prefill_decode_separately: bool = True

    notes: str = ""


def parse_memory_hierarchy(raw: Optional[dict[str, Any]]) -> MemoryHierarchyConfig:
    if not raw:
        return MemoryHierarchyConfig()

    # Backward compatibility with older YAML keys (hbm / dram / ssd).
    l2 = bool(raw.get("l2_hbm_tensor_path", raw.get("hbm_enabled", True)))
    host = bool(raw.get("host_dram_staging", raw.get("dram_staging", False)))
    pinned = bool(raw.get("host_dram_pinned", raw.get("dram_pinned", False)))
    ssd = raw.get("ssd_spill_path", raw.get("ssd_tier_path"))
    l3 = bool(raw.get("l3_dram_chunk_recall", False))
    r = int(raw.get("l3_recall_top_r", raw.get("l3_recall_r", 0)))

    return MemoryHierarchyConfig(
        l1_onchip_tiles=bool(raw.get("l1_onchip_tiles", True)),
        l2_hbm_tensor_path=l2,
        l3_dram_chunk_recall=l3,
        l3_recall_top_r=r,
        host_dram_staging=host,
        host_dram_pinned=pinned,
        ssd_spill_path=ssd,
        measure_prefill_decode_separately=bool(raw.get("measure_prefill_decode_separately", True)),
        notes=str(raw.get("notes", "")),
    )


def summarize_for_log(cfg: MemoryHierarchyConfig) -> str:
    parts = [
        f"L1_onchip={cfg.l1_onchip_tiles}",
        f"L2_HBM={cfg.l2_hbm_tensor_path}",
        f"L3_DRAM_recall={cfg.l3_dram_chunk_recall}",
        f"L3_R={cfg.l3_recall_top_r}",
        f"host_DRAM={cfg.host_dram_staging}",
        f"pinned={cfg.host_dram_pinned}",
        f"ssd={cfg.ssd_spill_path!r}",
        f"prefill_decode_split={cfg.measure_prefill_decode_separately}",
    ]
    if cfg.notes:
        parts.append(f"notes={cfg.notes!r}")
    return "[memory_hierarchy] " + " ".join(parts)


def warn_if_incomplete_tiers(cfg: MemoryHierarchyConfig) -> Optional[str]:
    """Return a warning string only when configuration looks inconsistent with paper tiers."""
    msgs: list[str] = []
    if not cfg.l2_hbm_tensor_path:
        msgs.append("L2 HBM path disabled; not comparable to paper GPU experiments.")
    if cfg.l3_dram_chunk_recall and cfg.l3_recall_top_r <= 0:
        msgs.append("L3 recall enabled but l3_recall_top_r<=0; set R (e.g. 8 per paper).")
    if cfg.l3_dram_chunk_recall and not (cfg.host_dram_staging or cfg.ssd_spill_path):
        msgs.append("L3 DRAM recall may need host_dram_staging or ssd_spill_path for buffers.")
    if not cfg.l1_onchip_tiles:
        msgs.append("L1 on-chip tiled path off; may not match FlashAttention-style throughput.")
    if not msgs:
        return None
    return "memory_hierarchy: " + " ".join(msgs)
