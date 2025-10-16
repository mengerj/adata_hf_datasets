#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a top-N "relevant genes" panel from HPA consensus RNA (50 tissues),
ranking by across-tissue variance + tissue-specificity (Tau) + mean, while
keeping protein_coding and selected lncRNA biotypes.

Overview
--------
This script generates a curated gene panel by:
1. Downloading Human Protein Atlas (HPA) tissue expression data and GENCODE annotations
2. Filtering genes by biotype (protein_coding, lincRNA, antisense, processed_transcript)
3. Scoring genes using a composite metric: α×variance + β×Tau + γ×mean expression
4. Optionally force-including curated cell markers from PanglaoDB and/or CellMarker databases
5. Selecting the top N genes by score

Usage in the Pipeline
---------------------
The **gs10k embedder** uses a 10,000-gene panel generated with this script using:
- Default scoring weights (α=0.5, β=0.4, γ=0.1)
- Both marker databases enabled: --include-cellmarker --include-panglaodb
- Detection threshold: nTPM >= 1.0 in at least 5 tissues

This ensures the gene panel includes:
- Genes with high variance and tissue-specificity (important for distinguishing cell types)
- Curated cell-type markers from established databases (ensures canonical markers are present)
- Sufficient mean expression (avoids extremely rare genes)

Marker Database Options
-----------------------
**--include-cellmarker**
  Downloads and integrates markers from the CellMarker database, a manually curated resource
  of cell markers for various cell types and tissues in human and mouse.
  Source: http://xteam.xbio.top/CellMarker/

  These markers are force-included in the final panel (if they pass detection filters),
  ensuring that well-established cell-type markers are present even if they don't score
  highly in the composite ranking.

**--include-panglaodb**
  Downloads and integrates markers from PanglaoDB, a database of cell-type markers
  compiled from single-cell RNA-seq experiments across multiple tissues and organisms.
  Source: https://panglaodb.se/

  Like CellMarker, these are force-included to ensure comprehensive cell-type coverage.

GENCODE GTF gene_id has version suffixes (e.g., ENSG00000141510.18). HPA "Gene"
uses unversioned Ensembl IDs (e.g., ENSG00000141510). We now normalize (strip
versions) on BOTH sides before intersecting, to avoid an empty matrix.

Data Sources
------------
- GENCODE GTF (gene annotations): https://www.gencodegenes.org/pages/data_format.html
- HPA consensus RNA (tissue expression): https://www.proteinatlas.org/about/download
- CellMarker (cell-type markers): http://xteam.xbio.top/CellMarker/
- PanglaoDB (cell-type markers): https://panglaodb.se/

Examples
--------
# Basic usage - top 10k genes without marker databases:
$ python get_genes.py --outdir data --top-n 10000 --verbose

# Recommended usage for gs10k embedder (includes marker databases):
$ python get_genes.py --outdir data --top-n 10000 --min-tissues 5 \
    --detect-threshold 1.0 --include-cellmarker --include-panglaodb --verbose

# Custom scoring weights (emphasize tissue-specificity):
$ python get_genes.py --outdir data --top-n 5000 --alpha 0.3 --beta 0.6 --gamma 0.1 --verbose

# Use existing downloads (skip re-downloading):
$ python get_genes.py --outdir data --top-n 10000 --skip-downloads --verbose

Output Files
------------
- top{N}_relevant_genes.csv: Full panel with scores and statistics
- top{N}_ensembl_ids.txt: Unversioned Ensembl gene IDs (one per line, for filtering)

Dependencies
------------
pandas, numpy, requests, tqdm
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ---------------- Defaults (override via env or CLI) ----------------

HPA_CONSENSUS_URL = os.getenv(
    "HPA_CONSENSUS_URL",
    "https://www.proteinatlas.org/download/tsv/rna_tissue_consensus.tsv.zip",
)
GENCODE_GTF_URL = os.getenv(
    "GENCODE_GTF_URL",
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49/gencode.v49.annotation.gtf.gz",
)

# Optional marker sources (CellMarker TXT; PanglaoDB TSV/TSV.GZ mirrors)
CELLMARKER_TXT_URLS: Tuple[str, ...] = (
    "https://bio-bigdata.hrbmu.edu.cn/CellMarker/download/Human_cell_markers.txt",
    # Standby/mirror announced by maintainers
    "https://xteam.xbio.top/CellMarker/download/Human_cell_markers.txt",
)
PANGLOADB_TSV_MIRRORS: Tuple[str, ...] = (
    "https://raw.githubusercontent.com/sumeetg23/10x-scRNA-Analysis/main/PanglaoDB_markers_27_Mar_2020.tsv.gz",
    "https://raw.githubusercontent.com/ImXman/MACA/master/MarkerDatabase/PanglaoDB_markers_27_Mar_2020.tsv",
)

DEFAULT_KEEP_BIOTYPES: Tuple[str, ...] = (
    "protein_coding",
    "lincRNA",
    "antisense",
    "processed_transcript",
)

EXCLUDE_MT_BIOTYPES: Tuple[str, ...] = ("rRNA", "rRNA_pseudogene", "Mt_rRNA", "Mt_tRNA")


# ---------------- Utilities ----------------


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def stream_download(url: str, dest: Path, chunk_size: int = 1 << 20) -> Optional[Path]:
    """
    Stream-download a file with progress.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Destination path.
    chunk_size : int
        Bytes per chunk (default 1 MiB).

    Returns
    -------
    Optional[Path]
        Path if successful; None if failed.
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading: {url}")
        with requests.get(url, stream=True, timeout=90) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            with (
                open(dest, "wb") as f,
                tqdm(
                    total=total if total > 0 else None,
                    unit="B",
                    unit_scale=True,
                    desc=dest.name,
                ) as pbar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        logging.info(f"Saved: {dest}")
        return dest
    except Exception as e:
        logging.warning(f"Download failed for {url}: {e}")
        return None


def _strip_ens_version(x: str) -> str:
    """
    Strip Ensembl version suffix after the first dot.

    Examples
    --------
    >>> _strip_ens_version("ENSG00000141510.18")
    'ENSG00000141510'
    >>> _strip_ens_version("ENSG00000141510")
    'ENSG00000141510'
    """
    if not isinstance(x, str):
        return x
    return x.split(".", 1)[0]


def parse_gtf_genes(gtf_gz: Path) -> pd.DataFrame:
    """
    Read GENCODE GTF (genes only) and extract metadata. Also provide a column
    with version-stripped Ensembl gene IDs for joining against HPA.

    Parameters
    ----------
    gtf_gz : Path
        Path to compressed GTF (.gtf.gz).

    Returns
    -------
    pandas.DataFrame
        Columns:
        - seqname
        - gene_type
        - gene_name
        - ensembl_gene_id         (possibly versioned)
        - ensembl_gene_id_novers  (version stripped: 'ENSG...')
    """
    cols = [
        "seqname",
        "source",
        "feature",
        "start",
        "end",
        "score",
        "strand",
        "frame",
        "attribute",
    ]
    with gzip.open(gtf_gz, "rt") as fh:
        df = pd.read_csv(
            fh,
            sep="\t",
            comment="#",
            names=cols,
            dtype={"seqname": "string", "feature": "string", "attribute": "string"},
            low_memory=False,
        )

    df = df[df["feature"] == "gene"].copy()

    def get_attr(attr: str, key: str) -> Optional[str]:
        m = re.search(rf'{key} "([^"]+)"', attr)
        return m.group(1) if m else None

    df["gene_name"] = df["attribute"].apply(lambda a: get_attr(a, "gene_name"))
    df["gene_type"] = df["attribute"].apply(lambda a: get_attr(a, "gene_type"))
    df["ensembl_gene_id"] = df["attribute"].apply(lambda a: get_attr(a, "gene_id"))
    df["ensembl_gene_id_novers"] = df["ensembl_gene_id"].map(_strip_ens_version)

    out = df[
        [
            "seqname",
            "gene_type",
            "gene_name",
            "ensembl_gene_id",
            "ensembl_gene_id_novers",
        ]
    ].dropna()
    logging.info(f"GTF genes parsed: {len(out):,}")
    return out


def load_hpa_expression_long(hpa_zip: Path) -> pd.DataFrame:
    """
    Load HPA consensus RNA (gene-level) long table and standardize columns.

    Returns a DataFrame with columns:
    - 'Gene'         (unversioned Ensembl gene ID)
    - 'Tissue'
    - 'nTPM'

    Parameters
    ----------
    hpa_zip : Path
        Path to 'rna_tissue_consensus.tsv.zip'.

    Returns
    -------
    pandas.DataFrame
        Long-form table with standardized columns.
    """
    df = pd.read_csv(hpa_zip, sep="\t", compression="zip", low_memory=False)
    # Expected columns per HPA docs: Gene, Tissue, nTPM
    # (older mirrors or pre-24.0 may have slightly different casing; be tolerant)
    rename = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "gene":
            rename[c] = "Gene"
        elif lc == "tissue":
            rename[c] = "Tissue"
        elif lc == "ntpm":
            rename[c] = "nTPM"
    if rename:
        df = df.rename(columns=rename)

    for c in ("Gene", "Tissue", "nTPM"):
        if c not in df.columns:
            raise RuntimeError(
                f"Expected column '{c}' not found in HPA file; got: {list(df.columns)[:12]}…"
            )

    # Normalize Ensembl IDs here as well (defensive)
    df["Gene"] = df["Gene"].astype(str).map(_strip_ens_version)

    logging.info(
        f"HPA rows: {len(df):,} | genes: {df['Gene'].nunique():,} | tissues: {df['Tissue'].nunique():,}"
    )
    return df[["Gene", "Tissue", "nTPM"]].copy()


def load_cellmarker_symbols(txt_path: Path) -> Set[str]:
    """
    Load CellMarker Human markers (TXT) and return a set of HGNC gene symbols.

    The TXT typically has a 'geneSymbol' column, but sometimes includes 'geneID'
    with comma-separated symbols. Parse both conservatively.
    """
    try:
        df = pd.read_csv(txt_path, sep="\t", dtype=str, low_memory=False)
        symbols: Set[str] = set()
        if "geneSymbol" in df.columns:
            symbols |= set(df["geneSymbol"].dropna().astype(str))
        if "geneID" in df.columns:
            for s in df["geneID"].dropna().astype(str):
                for g in re.split(r"[,\s;]+", s):
                    g = g.strip()
                    if g:
                        symbols.add(g)
        logging.info(f"CellMarker symbols loaded: {len(symbols):,}")
        return symbols
    except Exception as e:
        logging.warning(f"Failed to parse CellMarker markers from {txt_path}: {e}")
        return set()


def load_panglaodb_symbols(path: Path) -> Set[str]:
    """
    Load PanglaoDB markers TSV/TSV.GZ and return a set of HGNC symbols.
    """
    try:
        comp = "infer" if path.suffix == ".gz" else None
        df = pd.read_csv(path, sep="\t", compression=comp, dtype=str, low_memory=False)
        # Column can be 'official gene symbol' or 'gene' depending on mirror
        cols = [
            c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()
        ] or [c for c in df.columns if c.lower() in {"gene", "symbol"}]
        if not cols:
            cols = [df.columns[0]]
        symbols = set(df[cols[0]].dropna().astype(str))
        logging.info(f"PanglaoDB symbols loaded: {len(symbols):,}")
        return symbols
    except Exception as e:
        logging.warning(f"Failed to parse PanglaoDB markers from {path}: {e}")
        return set()


def tau(values: np.ndarray) -> float:
    """
    Compute Yanai’s tissue-specificity Tau in [0, 1]; higher = more specific.

    Parameters
    ----------
    values : np.ndarray
        Non-negative expression vector across tissues.

    Returns
    -------
    float
        Tau score.

    Notes
    -----
    Tau = sum(1 - x_i_hat) / (n - 1), x_i_hat = x_i / max(x).
    Returns 0.0 if all zeros.
    """
    x = np.clip(values, 0, None).astype(float)
    if x.size == 0:
        return 0.0
    m = x.max()
    if m <= 0:
        return 0.0
    x = x / m
    n = x.size
    return float(((1.0 - x).sum()) / (n - 1))


def zscore(v: np.ndarray) -> np.ndarray:
    """Standardize to zero mean and unit variance (eps to avoid div-by-zero)."""
    v = v.astype(float)
    return (v - v.mean()) / (v.std(ddof=0) + 1e-9)


def build_panel(
    gtf: pd.DataFrame,
    expr_long: pd.DataFrame,
    keep_biotypes: Sequence[str],
    min_tissues: int,
    detect_threshold: float,
    top_n: int,
    alpha: float,
    beta: float,
    gamma: float,
    marker_symbols: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Rank genes by composite (variance + Tau + mean), with detection & biotype filters.

    Parameters
    ----------
    gtf : pandas.DataFrame
        Output of `parse_gtf_genes`.
    expr_long : pandas.DataFrame
        Output of `load_hpa_expression_long` (Gene, Tissue, nTPM).
    keep_biotypes : sequence of str
        Biotypes to retain (e.g., protein_coding, lincRNA, antisense, processed_transcript).
    min_tissues : int
        Require nTPM ≥ detect_threshold in at least this many tissues.
    detect_threshold : float
        Detection cutoff (nTPM).
    top_n : int
        Target size of the panel.
    alpha, beta, gamma : float
        Weights for variance, Tau, and mean respectively.

    Returns
    -------
    pandas.DataFrame
        Columns: ['gene_name','ensembl_gene_id','mean_ntpm','var_ntpm','tau','score'].
    """
    # Keep relevant biotypes & exclude rRNA/MtRNA classes
    base = gtf[
        (gtf["gene_type"].isin(keep_biotypes))
        & (~gtf["gene_type"].isin(EXCLUDE_MT_BIOTYPES))
    ].copy()

    # Pivot HPA expression to (gene x tissue)
    mat = expr_long.groupby(["Gene", "Tissue"])["nTPM"].max().unstack(fill_value=0.0)

    # Intersect by *unversioned* Ensembl IDs
    idx_inter = mat.index.intersection(base["ensembl_gene_id_novers"])
    if len(idx_inter) == 0:
        raise RuntimeError(
            "No overlap between HPA 'Gene' IDs and GTF Ensembl IDs after version stripping. "
            "Check that you used a human GENCODE GTF matching GRCh38 and the HPA consensus file."
        )

    mat = mat.loc[idx_inter]
    base = base.set_index("ensembl_gene_id_novers").loc[idx_inter].reset_index()

    # Compute stats
    arr = mat.values  # shape: (genes, tissues)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise RuntimeError(
            f"Expression matrix has shape {arr.shape}; cannot score genes."
        )

    mean_ntpm = arr.mean(axis=1)
    var_ntpm = arr.var(axis=1)
    detected_in = (arr >= detect_threshold).sum(axis=1)
    taus = np.apply_along_axis(tau, 1, arr)

    # Detection floor
    keep_mask = detected_in >= int(min_tissues)
    if not np.any(keep_mask):
        raise RuntimeError(
            "All genes were filtered by the detection floor. "
            "Try lowering --detect-threshold or --min-tissues."
        )

    mean_ntpm = mean_ntpm[keep_mask]
    var_ntpm = var_ntpm[keep_mask]
    taus = taus[keep_mask]
    kept_symbols = base.loc[keep_mask, "gene_name"].astype(str).values
    kept_full_ids = base.loc[keep_mask, "ensembl_gene_id"].astype(str).values

    # Composite score
    score = alpha * zscore(var_ntpm) + beta * zscore(taus) + gamma * zscore(mean_ntpm)

    df = (
        pd.DataFrame(
            {
                "ensembl_gene_id": kept_full_ids,
                "gene_name": kept_symbols,
                "mean_ntpm": mean_ntpm,
                "var_ntpm": var_ntpm,
                "tau": taus,
                "score": score,
            }
        )
        .dropna()
        .sort_values("score", ascending=False)
    )

    # Optionally force-include curated marker symbols by unioning them before truncation
    if marker_symbols:
        forced_mask = df["gene_name"].isin(marker_symbols)
        n_forced = int(forced_mask.sum())
        if n_forced > 0:
            logging.info(
                f"Force-including {n_forced:,} curated marker genes (by gene_name)"
            )
        df = pd.concat([df[forced_mask], df[~forced_mask]], axis=0)
        df = df.drop_duplicates(subset=["gene_name"], keep="first")

    return df.head(top_n).reset_index(drop=True)


# ---------------- CLI ----------------


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Build a top-N relevant gene panel using HPA consensus RNA (variance + Tau + mean)."
    )
    p.add_argument(
        "--outdir", type=Path, default=Path("data"), help="Output directory."
    )
    p.add_argument(
        "--top-n", type=int, default=10000, help="Panel size (default 10,000)."
    )
    p.add_argument(
        "--min-tissues",
        type=int,
        default=5,
        help="Require detection (nTPM >= threshold) in at least K tissues.",
    )
    p.add_argument(
        "--detect-threshold",
        type=float,
        default=1.0,
        help="Detection threshold (nTPM, default 1.0).",
    )
    p.add_argument("--alpha", type=float, default=0.5, help="Weight for variance.")
    p.add_argument("--beta", type=float, default=0.4, help="Weight for Tau.")
    p.add_argument("--gamma", type=float, default=0.1, help="Weight for mean.")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    p.add_argument(
        "--skip-downloads",
        action="store_true",
        help="Use existing files in outdir/raw.",
    )
    p.add_argument(
        "--include-cellmarker",
        action="store_true",
        help="Download & union CellMarker (Human) markers.",
    )
    p.add_argument(
        "--include-panglaodb",
        action="store_true",
        help="Download & union PanglaoDB markers.",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    setup_logging(args.verbose)
    raw = args.outdir / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # Paths
    hpa_zip = raw / Path(HPA_CONSENSUS_URL).name
    gtf_gz = raw / Path(GENCODE_GTF_URL).name
    cellmarker_txt = raw / "Human_cell_markers.txt"
    panglao_tsv = raw / "PanglaoDB_markers_27_Mar_2020.tsv.gz"

    # Downloads
    if not args.skip_downloads:
        stream_download(HPA_CONSENSUS_URL, hpa_zip)
        stream_download(GENCODE_GTF_URL, gtf_gz)
        if args.include_cellmarker and not cellmarker_txt.exists():
            for url in CELLMARKER_TXT_URLS:
                if stream_download(url, cellmarker_txt):
                    break
            if not cellmarker_txt.exists():
                logging.warning(
                    "CellMarker TXT could not be downloaded; continuing without it."
                )
        if args.include_panglaodb and not panglao_tsv.exists():
            for url in PANGLOADB_TSV_MIRRORS:
                if stream_download(url, panglao_tsv):
                    break
            if not panglao_tsv.exists():
                logging.warning(
                    "PanglaoDB markers TSV could not be downloaded; continuing without it."
                )
    else:
        logging.info("Skipping downloads (using existing raw files).")

    # Load inputs
    gtf = parse_gtf_genes(gtf_gz)
    hpa = load_hpa_expression_long(hpa_zip)

    # Build & write
    # Optional curated marker union (by gene_name)
    marker_syms: Set[str] = set()
    if args.include_cellmarker and cellmarker_txt.exists():
        marker_syms |= load_cellmarker_symbols(cellmarker_txt)
    if args.include_panglaodb and panglao_tsv.exists():
        marker_syms |= load_panglaodb_symbols(panglao_tsv)
    if marker_syms:
        logging.info(f"Total curated marker symbols: {len(marker_syms):,}")

    panel = build_panel(
        gtf=gtf,
        expr_long=hpa,
        keep_biotypes=DEFAULT_KEEP_BIOTYPES,
        min_tissues=args.min_tissues,
        detect_threshold=args.detect_threshold,
        top_n=args.top_n,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        marker_symbols=marker_syms if marker_syms else None,
    )

    out_csv = args.outdir / f"top{args.top_n}_relevant_genes.csv"
    panel.to_csv(out_csv, index=False)
    logging.info(f"Wrote: {out_csv}  rows={len(panel)}")
    logging.info("Preview:\n" + panel.head(10).to_string(index=False))

    # Also save a TXT with unversioned Ensembl gene IDs (one per line)
    ens_novers = (
        panel["ensembl_gene_id"]
        .astype(str)
        .map(_strip_ens_version)
        .dropna()
        .drop_duplicates()
    )
    out_txt = args.outdir / f"top{args.top_n}_ensembl_ids.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        for gid in ens_novers:
            f.write(f"{gid}\n")
    logging.info(f"Wrote: {out_txt}  rows={len(ens_novers)} (unversioned Ensembl IDs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
