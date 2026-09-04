import os
import glob
import datetime
import io
import re
import sys
import multiprocessing
import time
import platform
import zipfile
import json
from contextlib import contextmanager
import brim_provenance
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

APP_VERSION = "1.1.0"
APP_NAME = "BRIM"
APP_SUBTITLE = "Bulk RNA-seq Insight in Minutes"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from brim_enrichment import resolve_gene_set, run_overrepresentation
from brim_tf_networks import infer_tf_activity, load_collectri_network, load_dorothea_network
import seaborn as sns
# i18n
from i18n import LANGUAGE_OPTIONS, t, ui

import streamlit as st
try:
    from streamlit_plotly_events import plotly_events as _plotly_events
    _HAS_PLOTLY_EVENTS = True
except ImportError:
    _HAS_PLOTLY_EVENTS = False
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.gridspec import GridSpec

# Platform check
is_mac = platform.system() == "Darwin"

# ═══════════════════════════════════════════
# 0. PAGE CONFIG & THEME
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="BRIM",
    page_icon="🎩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for a "Premium" Mac-like feel
mac_css = """
<style>
    /* Global font: San Francisco / Apple style */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol" !important;
    }
    
    /* Font sizes */
    html, body, .stApp {
        font-size: 20px !important;
    }
    .stMarkdown p, .stMarkdown li, .stText p {
        font-size: 20px !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {
        font-size: 17px !important;
    }
    
    /* Tabs labels */
    button[data-baseweb="tab"] p, [data-testid="stTab"] {
        font-size: 18px !important;
        font-weight: 600 !important;
    }
    
    /* Button text */
    .stButton > button, .stDownloadButton > button {
        font-size: 17px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1) !important;
    }

    /* Target mode selection buttons specifically */
    [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
        font-size: 18px !important;
        padding-top: 16px !important;
        padding-bottom: 16px !important;
        border-radius: 12px !important;
    }
    
    /* st.metric labels and values */
    [data-testid="stMetricValue"] div {
        font-size: 1.6rem !important;
    }
    [data-testid="stMetricLabel"] p {
        font-size: 1.0rem !important;
    }
    
    /* selectbox / radio / slider labels */
    .stSelectbox label, .stRadio label, .stSlider label, [data-testid="stWidgetLabel"] p {
        font-size: 17px !important;
    }
    
    /* Card-like containers for metrics */
    [data-testid="stMetric"] {
        background-color: rgba(151, 166, 195, 0.08);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(0,0,0,0.05);
    }
</style>
"""
st.markdown(mac_css, unsafe_allow_html=True)

# — Hero Header
_, h_center, _ = st.columns([1, 2, 1])
with h_center:
    try:
        st.image("brim_logo.png", width=160)
    except Exception:
        st.markdown("<div style='font-size:4rem; text-align:center;'>🎩</div>",
                    unsafe_allow_html=True)
    st.markdown(f"""
<div style='text-align:center; padding:16px 0 32px 0;'>
  <div style='font-size:3rem; font-weight:800; line-height:1.2;'>{APP_NAME}</div>
  <div style='font-size:1.1rem; color:gray; margin-top:8px;'>{APP_SUBTITLE}</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════
# 1. CONSTANTS & MAPPINGS
# ═══════════════════════════════════════════
SPECIES_MAP = {
    "Mouse (mmu)": {
        "org": "mmu", "string_id": 10090, 
        "gene_sets_kegg": "KEGG_2019_Mouse",
        "gene_sets_go": "GO_Biological_Process_2021"
    },
    "Human (hsa)": {
        "org": "hsa", "string_id": 9606, 
        "gene_sets_kegg": "KEGG_2021_Human",
        "gene_sets_go": "GO_Biological_Process_2021"
    },
}
FONTS = ["Arial", "Helvetica", "Times New Roman", "DejaVu Sans", "Roboto", "Courier New", "Verdana", "Georgia", "Segoe UI", "sans-serif"]


MMCP_COUNTER_MARKERS = {
    "T_cells":               ["Cd3d", "Cd3e", "Cd3g", "Cd247", "Trac"],
    "CD8_T_cells":           ["Cd8a", "Cd8b1", "Gzmk", "Trgc2", "Eomes"],
    "Cytotoxic_lymphocytes": ["Prf1", "Gzma", "Gzmb", "Nkg7", "Klrk1"],
    "B_lineage":             ["Cd19", "Ms4a1", "Cd79a", "Cd79b", "Pax5"],
    "NK_cells":              ["Ncam1", "Klrb1c", "Klrd1", "Xcl1", "Gzmb"],
    "Monocytes":             ["Cd14", "Csf1r", "Fcgr3", "S100a8", "S100a9"],
    "Macrophages":           ["Adgre1", "Mrc1", "Cd68", "Itgam", "C1qa"],
    "Neutrophils":           ["Csf3r", "S100a8", "Ly6g", "Cxcr2", "Mpo"],
    "Mast_cells":            ["Kit", "Ms4a2", "Cpa3", "Tpsb2", "Hpgds"],
    "Dendritic_cells":       ["Itgax", "H2-Eb1", "Clec9a", "Siglech", "Ccr7"],
    "Fibroblasts":           ["Col1a1", "Col3a1", "Acta2", "Pdgfra", "Thy1"],
    "Endothelial_cells":     ["Pecam1", "Cdh5", "Kdr", "Tek", "Esam"],
}

# ── 外部リファレンスCSVの自動スキャン ──────────────────────────
_REF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "references")
_EXTERNAL_REFS = {}
_FAILED_REFS = []
if os.path.isdir(_REF_DIR):
    for _ref_path in sorted(glob.glob(os.path.join(_REF_DIR, "*.csv"))):
        _ref_name = os.path.splitext(os.path.basename(_ref_path))[0]
        try:
            _EXTERNAL_REFS[_ref_name] = pd.read_csv(_ref_path, index_col=0)
        except Exception as _ref_err:
            _FAILED_REFS.append((_ref_name, str(_ref_err)))
# Source: Petitprez et al., Genome Medicine 12, 86 (2020)
# DOI: 10.1186/s13073-020-00783-w
# License: GPL-3



# Results storage
if "enr_kegg"     not in st.session_state: st.session_state["enr_kegg"]     = None
if "enr_go"       not in st.session_state: st.session_state["enr_go"]       = None
if "gsea_results" not in st.session_state: st.session_state["gsea_results"] = None
if "tf_results"   not in st.session_state: st.session_state["tf_results"]   = None
if "tf_collectri" not in st.session_state: st.session_state["tf_collectri"] = None
if "tf_dorothea"  not in st.session_state: st.session_state["tf_dorothea"]  = None
if "ciber_results" not in st.session_state: st.session_state["ciber_results"] = None
if "fig_font_sz"  not in st.session_state: st.session_state["fig_font_sz"]  = 12
if "rna_input_files" not in st.session_state: st.session_state["rna_input_files"] = []
if "rna_id_mapping" not in st.session_state: st.session_state["rna_id_mapping"] = []
if "external_service_events" not in st.session_state: st.session_state["external_service_events"] = []
if "external_service_history" not in st.session_state: st.session_state["external_service_history"] = []

# ═══════════════════════════════════════════
# 2. SESSION STATE
# ═══════════════════════════════════════════
if "analysis_log" not in st.session_state: st.session_state["analysis_log"] = []
if "counts_df" not in st.session_state: st.session_state["counts_df"] = None
if "qc_filtered_df" not in st.session_state: st.session_state["qc_filtered_df"] = None
if "deg_results" not in st.session_state: st.session_state["deg_results"] = None
if "metadata" not in st.session_state: st.session_state["metadata"] = None
if "conditions" not in st.session_state: st.session_state["conditions"] = [] # BUG FIX: バグ④
if "last_contrast" not in st.session_state: st.session_state["last_contrast"] = ""
if "venn_deg_sets"  not in st.session_state: st.session_state["venn_deg_sets"]  = None
if "venn_v_sel"     not in st.session_state: st.session_state["venn_v_sel"]     = None
if "venn_enr_kegg"  not in st.session_state: st.session_state["venn_enr_kegg"]  = None
if "venn_enr_go"    not in st.session_state: st.session_state["venn_enr_go"]    = None
if "up_color"      not in st.session_state: st.session_state["up_color"]      = "#E64B35"
if "down_color"    not in st.session_state: st.session_state["down_color"]    = "#4DBBD5"
if "fig_width"     not in st.session_state: st.session_state["fig_width"]     = 800
if "fig_height"    not in st.session_state: st.session_state["fig_height"]    = 500
if "img_format"    not in st.session_state: st.session_state["img_format"]    = "png"
if "img_dpi"       not in st.session_state: st.session_state["img_dpi"]       = 300
if "enr_cmap"      not in st.session_state: st.session_state["enr_cmap"]      = "Viridis_r"
if "hm_cmap"       not in st.session_state: st.session_state["hm_cmap"]       = "RdBu_r"
if "selected_font" not in st.session_state: st.session_state["selected_font"] = "sans-serif"
if "lfc_t"          not in st.session_state: st.session_state["lfc_t"]          = 1.0
if "padj_t"         not in st.session_state: st.session_state["padj_t"]         = 0.05
if "deg_t"          not in st.session_state: st.session_state["deg_t"]          = (1.0, 0.05)


_DATA_RESULT_DEFAULTS = {
    "deg_results": None,
    "last_contrast": "",
    "batch_deg_results": {},
    "batch_deg_provenance": {},
    "enr_kegg": None,
    "enr_go": None,
    "gsea_results": None,
    "gsea_object": None,
    "tf_results": None,
    "tf_collectri": None,
    "tf_dorothea": None,
    "ciber_results": None,
    "venn_deg_sets": None,
    "venn_v_sel": None,
    "venn_enr_kegg": None,
    "venn_enr_go": None,
    "ia_results": None,
    "ia_all_term_results": {},
    "ia_coef_names": [],
    "ia_lasso_genes": [],
    "custom_gene_list": [],
    "lfc_meta_matrix": None,
}


def reset_data_results():
    """Clear every result derived from a count matrix before replacing it."""
    for key, default in _DATA_RESULT_DEFAULTS.items():
        st.session_state[key] = default.copy() if isinstance(default, (dict, list)) else default
    st.session_state["analysis_log"] = []


def has_data_results():
    """Return whether any analysis output is currently stored."""
    for key in _DATA_RESULT_DEFAULTS:
        value = st.session_state.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, str)) and len(value) == 0:
            continue
        return True
    return False


def reset_contrast_results():
    """Clear outputs that depend on the currently selected DEG contrast."""
    for key in (
        "enr_kegg", "enr_go", "gsea_results", "gsea_object",
        "tf_results", "tf_collectri", "tf_dorothea", "ciber_results",
        "custom_gene_list",
    ):
        st.session_state[key] = [] if key == "custom_gene_list" else None


def reset_threshold_dependent_results():
    """Clear cached selections/results whose gene membership uses DEG thresholds."""
    reset_contrast_results()
    for key in ("venn_deg_sets", "venn_v_sel", "venn_enr_kegg", "venn_enr_go"):
        st.session_state[key] = None


def log_analysis(action, details=""):
    st.session_state["analysis_log"].append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "action": action, "details": details})


def external_service_record(service, data_type, source=None):
    """Journal a lookup before execution; never discard it on input failure."""
    history = st.session_state["external_service_history"]
    event = {"event_id": len(history) + 1, "service": service, "data_type": data_type,
             "timestamp": datetime.datetime.now().astimezone().isoformat(),
             "source": dict(source) if source is not None else None,
             "input_outcome": "pending" if source is not None else "not_applicable",
             "lookup_outcome": "pending", "access": "pending", "requests": []}
    history.append(event)
    return event


@contextmanager
def service_input_attempt():
    """Reject uncommitted lookup associations even on st.stop or exceptions."""
    events = []
    try:
        yield events
    finally:
        for event in events:
            if event["input_outcome"] == "pending":
                event["input_outcome"] = "rejected"


def service_post(event, url, data):
    """Record every real HTTP attempt before sending, including failed chunks."""
    request = {"timestamp": datetime.datetime.now().astimezone().isoformat(),
               "outcome": "pending", "http_status": None}
    event["requests"].append(request)
    try:
        response = requests.post(url, data=data, timeout=30)
    except Exception as error:
        request.update(outcome="failed", error_type=type(error).__name__)
        raise
    request.update(http_status=response.status_code,
                   outcome="success" if response.status_code == 200 else "failed")
    return response, request


def service_lookup(event, cached_lookup, *args, **kwargs):
    """Keep cached results separate from actual requests in this session."""
    try:
        value, outcome = cached_lookup(*args, _event=event, **kwargs)
        event["lookup_outcome"] = outcome
        return value
    except Exception as error:
        event.update(lookup_outcome="failed", error_type=type(error).__name__)
        raise
    finally:
        event["access"] = "network" if event["requests"] else "cache"


def get_string_network_img(gene_list, species_id, limit=30, flavor="confidence", *, event=None):
    """Fetch the existing STRING image while retaining the lookup's outcome."""
    if event is None:
        event = external_service_record("string-db.org", "gene list")
    return service_lookup(event, _cached_string_network_img, gene_list, species_id, limit, flavor)

@st.cache_data(show_spinner=False, ttl=300)
def _cached_string_network_img(gene_list, species_id, limit, flavor, _event):
    url = "https://string-db.org/api/image/network"
    params = {
        "identifiers": "\r".join(gene_list[:limit]),
        "species": species_id,
        "add_white_nodes": 1,
        "network_flavor": flavor
    }
    try:
        res, _ = service_post(_event, url, params)
        return (res.content, "success") if res.status_code == 200 else (None, "failed")
    except requests.RequestException:
        return None, "failed"

# ═══════════════════════════════════════════
# 3. ANALYSIS HELPERS
# ═══════════════════════════════════════════
def run_online_mapping(id_list, species_id, *, event=None):
    """Map IDs with the existing cache, preserving request and lookup outcomes."""
    if event is None:
        event = external_service_record("mygene.info", "gene IDs")
    return service_lookup(event, _cached_online_mapping, id_list, species_id)


@st.cache_data(show_spinner=False, ttl=300)
def _cached_online_mapping(id_list, species_id, _event):
    mapped_dict = {}
    failed = False
    for i in range(0, len(id_list), 1000):
        chunk = id_list[i:i+1000]
        request = None
        try:
            res, request = service_post(_event, "https://mygene.info/v3/query", {'q':",".join(chunk),'scopes':'ensembl.gene,entrezgene,refseq,uniprot','species':species_id,'fields':'symbol'})
            if res.status_code == 200:
                payload = res.json()
                if not isinstance(payload, list):
                    raise ValueError("Mapping response must be a list.")
                for item in payload:
                    if (not isinstance(item, dict) or not isinstance(item.get("query"), str)
                            or item["query"] not in chunk
                            or ("symbol" in item and (not isinstance(item["symbol"], str) or not item["symbol"]))):
                        raise ValueError("Invalid mapping response entry.")
                    if 'symbol' in item:
                        mapped_dict[item['query']] = item['symbol']
            else:
                failed = True
        except (requests.RequestException, ValueError, TypeError) as error:
            failed = True
            if request is not None:
                request.update(outcome="failed", error_type=type(error).__name__)
        if i + 1000 < len(id_list):
            time.sleep(0.3)
    if not failed and len(mapped_dict) == len(set(id_list)):
        outcome = "success"
    elif mapped_dict:
        outcome = "partial"
    else:
        outcome = "failed" if failed else "unmapped"
    return mapped_dict, outcome



def normalize_counts(counts_df, method="log1p", gene_lengths=None):
    """Apply selected normalization to count matrix."""
    if counts_df is None or counts_df.empty:
        raise ValueError("No genes remain for normalization.")
    zero_library_samples = counts_df.columns[counts_df.sum(axis=0) <= 0].tolist()
    if zero_library_samples:
        raise ValueError(
            "Samples with zero total counts cannot be normalized: "
            + ", ".join(map(str, zero_library_samples))
        )
    if method == "log1p":
        # log1p(CPM): 可視化・探索的解析向けの簡便法。DEG解析には raw counts を用い、相関解析やPCAではデータによってVSTがより適切な場合があります。
        lib_size = counts_df.sum(axis=0)
        cpm = counts_df.div(lib_size, axis=1) * 1e6
        return np.log1p(cpm)
    elif method == "CPM":
        lib_size = counts_df.sum(axis=0)
        return (counts_df.div(lib_size, axis=1) * 1e6)
    elif method == "TPM":
        import streamlit as st

        if gene_lengths is not None:
            try:
                gene_lengths = prepare_gene_lengths(gene_lengths)
            except ValueError as _length_error:
                st.error(ui("Invalid gene length data: {error}", lang, error=_length_error))
                st.stop()
            common_genes = counts_df.index.intersection(gene_lengths.index)

            if len(common_genes) == 0:
                st.error(ui("⛔ Gene IDs in counts data and gene length data do not match. TPM calculation was stopped.", lang))
                st.stop()

            # 遺伝子の欠落を検出して警告
            _n_total_genes = len(counts_df.index)
            _n_common = len(common_genes)
            _n_dropped = _n_total_genes - _n_common
            _drop_rate = _n_dropped / _n_total_genes if _n_total_genes > 0 else 0
            if _drop_rate >= 0.5:
                st.warning(ui("⚠️ TPM: Only {matched:,} / {total:,} genes matched gene-length data; {dropped:,} ({rate:.0%}) were excluded. Check that gene ID formats match.", lang,
                              matched=_n_common, total=_n_total_genes, dropped=_n_dropped, rate=_drop_rate))
            elif _n_dropped > 0:
                st.info(ui("ℹ️ TPM: Using {matched:,} / {total:,} genes ({dropped:,} excluded because length data were unavailable).", lang,
                           matched=_n_common, total=_n_total_genes, dropped=_n_dropped))

            c_df = counts_df.loc[common_genes]
            lengths_kb = gene_lengths.loc[common_genes] / 1000.0
            rpk = c_df.div(lengths_kb, axis=0)

            scaling_factor = rpk.sum(axis=0) / 1e6
            return rpk.div(scaling_factor, axis=1)

        else:
            st.error(ui("⛔ Gene length data is required for TPM calculation. Please provide gene length data in the Upload tab.", lang))
            st.stop()
    elif method == "VST":
        try:
            from pydeseq2.preprocessing import vst_fit_transform
            return pd.DataFrame(
                vst_fit_transform(counts_df.T.values),
                index=counts_df.columns,
                columns=counts_df.index
            ).T
        except Exception as _vst_err:
            import streamlit as st
            st.warning(ui("⚠️ VST failed ({error}). Falling back to log1p(CPM). Check that PyDESeq2 is installed correctly.", lang,
                          error=_vst_err))
            lib_size = counts_df.sum(axis=0)
            cpm = counts_df.div(lib_size, axis=1) * 1e6
            return np.log1p(cpm)
    lib_size = counts_df.sum(axis=0)
    cpm = counts_df.div(lib_size, axis=1) * 1e6
    return np.log1p(cpm)


def prepare_count_matrix(raw_df):
    """Validate raw count input without silently changing invalid values."""
    if raw_df is None or raw_df.empty or raw_df.shape[1] == 0:
        raise ValueError("The count matrix must contain at least one gene and one sample.")
    if raw_df.columns.duplicated().any():
        duplicates = raw_df.columns[raw_df.columns.duplicated()].astype(str).tolist()
        raise ValueError("Duplicate sample names: " + ", ".join(duplicates))
    if raw_df.index.isna().any() or (raw_df.index.astype(str).str.strip() == "").any():
        raise ValueError("Gene names must not be empty.")

    numeric = raw_df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        invalid_count = int(numeric.isna().sum().sum())
        raise ValueError(f"{invalid_count} missing or non-numeric count value(s) detected.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Infinite count values are not allowed.")
    if (values < 0).any():
        raise ValueError(f"{int((values < 0).sum())} negative count value(s) detected.")
    if not np.allclose(values, np.rint(values), rtol=0, atol=1e-9):
        raise ValueError("Decimal values were detected. Raw integer counts are required.")

    counts_df = numeric.astype(np.int64)
    if counts_df.index.duplicated().any():
        counts_df = counts_df.groupby(level=0, sort=False).sum()
    if counts_df.empty:
        raise ValueError("The count matrix contains no valid genes.")
    zero_library_samples = counts_df.columns[counts_df.sum(axis=0) <= 0].astype(str).tolist()
    if zero_library_samples:
        raise ValueError("Samples with zero total counts: " + ", ".join(zero_library_samples))
    return counts_df


def read_count_sample_names(file_obj, sep):
    """Read and validate only the header row of a count matrix."""
    file_obj.seek(0)
    header = pd.read_csv(file_obj, sep=sep, header=None, nrows=1, dtype=str)
    file_obj.seek(0)
    if header.shape[1] < 2:
        raise ValueError("The count matrix must contain a gene column and at least one sample column.")
    sample_names = header.iloc[0, 1:].fillna("").astype(str).str.strip()
    if (sample_names == "").any():
        raise ValueError("Sample names must not be empty.")
    if sample_names.duplicated().any():
        duplicates = sample_names[sample_names.duplicated()].tolist()
        raise ValueError("Duplicate sample names: " + ", ".join(duplicates))
    return sample_names.tolist()


def read_count_matrix_file(file_obj, sep):
    """Read a count file while preserving detection of duplicate sample headers."""
    read_count_sample_names(file_obj, sep)
    frame = pd.read_csv(file_obj, sep=sep, index_col=0)
    file_obj.seek(0)
    return frame


def prepare_gene_lengths(raw_lengths):
    """Validate gene lengths used for TPM; every value must be positive and finite."""
    if raw_lengths is None or len(raw_lengths) == 0:
        raise ValueError("Gene length data is empty.")
    if raw_lengths.index.duplicated().any():
        duplicates = raw_lengths.index[raw_lengths.index.duplicated()].astype(str).tolist()
        raise ValueError("Duplicate gene IDs: " + ", ".join(duplicates[:10]))
    if raw_lengths.index.isna().any() or (raw_lengths.index.astype(str).str.strip() == "").any():
        raise ValueError("Gene IDs must not be empty.")
    numeric = pd.to_numeric(raw_lengths, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError(f"{int(np.isnan(values).sum())} missing or non-numeric length value(s) detected.")
    if not np.isfinite(values).all():
        raise ValueError("Infinite gene lengths are not allowed.")
    if (values <= 0).any():
        raise ValueError(f"{int((values <= 0).sum())} zero or negative gene length value(s) detected.")
    return numeric.astype(float)


def active_counts_df():
    """Return filtered counts when present, without truth-testing a DataFrame."""
    filtered = st.session_state.get("qc_filtered_df")
    return filtered if filtered is not None else st.session_state.get("counts_df")

# ── 共通遺伝子リスト（Single / Multi Study 両方で使用） ──────────────
_IMMUNE_GENES = [
    "Tnf","Il6","Il1b","Il10","Il4","Il13","Ifng","Tgfb1","Ccl2","Ccl5",
    "Cxcl1","Cxcl10","Stat1","Stat3","Stat6","Nfkb1","Irf3","Irf7","Tlr4","Myd88",
    "Cd4","Cd8a","Foxp3","Gata3","Tbx21","Rorc","Il17a","Il22","Il33","Il25",
    "Tslp","Il31","Il5","Csf2","Vegfa","Mmp9","Mmp2","Timp1","Col1a1","Col3a1",
    "Fn1","Vim","Cdh1","Ocln","Cldn1","Krt1","Krt10","Flg","Lor","Ivl"
]
_BARRIER_GENES = [
    "Flg","Lce1a","Lce1b","Lce2a","Krt2","Krt5","Krt14","Krt16",
    "Dsg1","Dsg3","Dsp","Pkp1","Gja1","Aqp3","Smpd1","Cers3","Elovl4","Fa2h",
    "Abca12","Cldn4","Cldn7","Cldn11","Tjp1","Ocln","Cdh2","Itgb4","Itga6",
    "Lamb3","Lamc2","Lama3"
]
_SIGNAL_GENES = [
    "Egfr","Erbb2","Fgfr1","Pdgfra","Igf1r","Insr","Met",
    "Akt1","Akt2","Pten","Mtor","Mapk1","Mapk3","Mapk8","Mapk14","Mapk9",
    "Pik3ca","Pik3r1","Kras","Hras","Nras","Braf","Raf1","Map2k1",
    "Jak1","Jak2","Tyk2","Socs1","Socs3","Ptpn11","Grb2","Sos1","Shc1",
    "Plcg1","Prkca","Prkcb","Prkcd","Calm1","Camk2a","Creb1","Jun","Fos",
    "Myc","Tp53","Rb1","Cdkn1a","Cdkn2a","Bcl2","Bax"
]
_EXTRA_GENES = [f"Gene{str(i).zfill(3)}" for i in range(1, 371)]
_ALL_GENES = _IMMUNE_GENES + _BARRIER_GENES + _SIGNAL_GENES + _EXTRA_GENES  # 計500遺伝子

@st.cache_data(show_spinner=False)
def generate_sample_data():
    try:
        counts_df = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "sample_data/single_study_counts.csv"),
            index_col=0
        )
        metadata = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "sample_data/single_study_metadata.csv"),
            index_col=0
        )
        return counts_df, metadata
    except FileNotFoundError:
        np.random.seed(42)
        genes = _ALL_GENES  # 500遺伝子・Gene Symbol直接使用
        n_genes = len(genes)
        samples = [f"Ctrl_{i}" for i in range(1,5)] + \
                  [f"TrtA_{i}" for i in range(1,5)] + \
                  [f"TrtB_{i}" for i in range(1,5)]  # 12サンプル・3群各4
        
        base_expr = np.random.negative_binomial(n=20, p=0.05, size=(n_genes, 12)).astype(float)
        
        # TrtA: 免疫系遺伝子25〜49をup（index 25-49）
        base_expr[25:50, 4:8] = base_expr[25:50, 4:8] * np.random.uniform(3.0, 8.0, size=(25, 4))
        # TrtA: 免疫系遺伝子0〜24をdown
        base_expr[0:25, 4:8]  = base_expr[0:25, 4:8]  * np.random.uniform(0.1, 0.4, size=(25, 4))
        # TrtB: 免疫系遺伝子0〜49をup
        base_expr[0:50, 8:12] = base_expr[0:50, 8:12] * np.random.uniform(2.0, 5.0, size=(50, 4))
        # TrtB: バリア遺伝子50〜79をdown
        base_expr[50:80, 8:12] = base_expr[50:80, 8:12] * np.random.uniform(0.1, 0.4, size=(30, 4))
        # TrtA: シグナル遺伝子80〜129をup
        base_expr[80:130, 4:8] = base_expr[80:130, 4:8] * np.random.uniform(4.0, 10.0, size=(50, 4))
        
        base_expr = np.clip(base_expr, 0, None).astype(int)
        counts_df = pd.DataFrame(base_expr, index=genes, columns=samples)
        meta_dict = {s: ("Control" if "Ctrl" in s else ("Treatment_A" if "TrtA" in s else "Treatment_B")) for s in samples}
        metadata = pd.DataFrame.from_dict(meta_dict, orient="index", columns=["condition"])
        return counts_df, metadata


@st.cache_data(show_spinner=False)
def generate_multi_study_sample_data():
    try:
        base_path = os.path.dirname(__file__)
        pdf_a = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_atopic_counts.csv"), index_col=0)
        meta_a = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_atopic_metadata.csv"), index_col=0)
        pdf_p = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_psoriasis_counts.csv"), index_col=0)
        meta_p = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_psoriasis_metadata.csv"), index_col=0)
        pdf_e = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_aew_counts.csv"), index_col=0)
        meta_e = pd.read_csv(os.path.join(base_path, "sample_data/multi_study_aew_metadata.csv"), index_col=0)
        
        return {
            "Atopic": {"counts": pdf_a, "metadata": meta_a},
            "Psoriasis": {"counts": pdf_p, "metadata": meta_p},
            "AEW": {"counts": pdf_e, "metadata": meta_e}
        }
    except FileNotFoundError:
        np.random.seed(123)
        genes = _ALL_GENES  # 500遺伝子・全Study共通
        n_genes = len(genes)

        def _idx(name):
            try: return genes.index(name)
            except ValueError: return None

        result = {}

        # ── Study 1: Atopic ──────────────────────────────────────────
        sA = [f"Atopic_Ctrl_{i}" for i in range(1,4)] + [f"Atopic_Disease_{i}" for i in range(1,4)]
        exA = np.random.negative_binomial(n=20, p=0.05, size=(n_genes, 6)).astype(float)
        for g in ["Il4","Il13","Il5","Il33","Tslp","Il31"]:
            idx = _idx(g)
            if idx is not None: exA[idx, 3:6] *= np.random.uniform(5.0, 10.0, size=3)
        for g in ["Flg","Lor","Ivl","Krt1","Krt10","Cldn1","Ocln"]:
            idx = _idx(g)
            if idx is not None: exA[idx, 3:6] *= np.random.uniform(0.1, 0.3, size=3)
        for g in ["Tnf","Il6","Stat3","Nfkb1"]:
            idx = _idx(g)
            if idx is not None: exA[idx, 3:6] *= np.random.uniform(2.0, 4.0, size=3)
        exA = np.clip(exA, 0, None).astype(int)
        cdfA = pd.DataFrame(exA, index=genes, columns=sA)
        metaA = pd.DataFrame(
            {"condition": ["Control"]*3 + ["Atopic_Disease"]*3, "batch": "Atopic"},
            index=sA
        )
        result["Atopic"] = {"counts": cdfA, "metadata": metaA}

        # ── Study 2: Psoriasis ───────────────────────────────────────
        sP = [f"Psori_Ctrl_{i}" for i in range(1,4)] + [f"Psori_Disease_{i}" for i in range(1,4)]
        exP = np.random.negative_binomial(n=20, p=0.05, size=(n_genes, 6)).astype(float)
        for g in ["Il17a","Il22","Tnf","Il6","Cxcl1","Cxcl10"]:
            idx = _idx(g)
            if idx is not None: exP[idx, 3:6] *= np.random.uniform(5.0, 10.0, size=3)
        for g in ["Flg","Lor","Krt1"]:
            idx = _idx(g)
            if idx is not None: exP[idx, 3:6] *= np.random.uniform(0.2, 0.4, size=3)
        for g in ["Stat1","Stat3","Nfkb1","Irf3"]:
            idx = _idx(g)
            if idx is not None: exP[idx, 3:6] *= np.random.uniform(3.0, 6.0, size=3)
        exP = np.clip(exP, 0, None).astype(int)
        cdfP = pd.DataFrame(exP, index=genes, columns=sP)
        metaP = pd.DataFrame(
            {"condition": ["Control"]*3 + ["Psoriasis_Disease"]*3, "batch": "Psoriasis"},
            index=sP
        )
        result["Psoriasis"] = {"counts": cdfP, "metadata": metaP}

        # ── Study 3: AEW ─────────────────────────────────────────────
        sE = [f"AEW_Ctrl_{i}" for i in range(1,4)] + [f"AEW_Treated_{i}" for i in range(1,4)]
        exE = np.random.negative_binomial(n=20, p=0.05, size=(n_genes, 6)).astype(float)
        for g in ["Flg","Lor","Ivl","Cldn1","Ocln","Krt1","Krt10","Dsg1"]:
            idx = _idx(g)
            if idx is not None: exE[idx, 3:6] *= np.random.uniform(0.1, 0.3, size=3)
        for g in ["Il4","Il33","Tslp","Tnf","Il6"]:
            idx = _idx(g)
            if idx is not None: exE[idx, 3:6] *= np.random.uniform(2.0, 5.0, size=3)
        for g in ["Cers3","Elovl4","Abca12","Aqp3"]:
            idx = _idx(g)
            if idx is not None: exE[idx, 3:6] *= np.random.uniform(0.2, 0.5, size=3)
        exE = np.clip(exE, 0, None).astype(int)
        cdfE = pd.DataFrame(exE, index=genes, columns=sE)
        metaE = pd.DataFrame(
            {"condition": ["Control"]*3 + ["AEW_Treated"]*3, "batch": "AEW"},
            index=sE
        )
        result["AEW"] = {"counts": cdfE, "metadata": metaE}

        return result

def run_deg(counts_df, metadata, ref_condition, test_condition, n_cpus=1):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    # A contrast must be fitted only from its two selected groups.  This is
    # especially important in multi-study mode, where unrelated studies must
    # not enter a within-study dispersion model without a batch term.
    contrast_metadata = metadata.loc[
        metadata["condition"].isin([ref_condition, test_condition])
    ].copy()
    if contrast_metadata["condition"].nunique() != 2:
        raise ValueError("Both contrast groups must be present in metadata.")
    missing_samples = contrast_metadata.index.difference(counts_df.columns)
    if len(missing_samples):
        raise ValueError("Metadata samples missing from count matrix: " + ", ".join(map(str, missing_samples)))
    count_matrix = counts_df.loc[:, contrast_metadata.index].T

    dds = DeseqDataSet(
        counts=count_matrix,
        metadata=contrast_metadata,
        design="~condition",
        refit_cooks=True,
        n_cpus=n_cpus
    )
    dds.deseq2()

    stat_res = DeseqStats(
        dds,
        contrast=["condition", test_condition, ref_condition],
        n_cpus=n_cpus
    )
    stat_res.summary()
    res = stat_res.results_df.copy()
    res["padj_is_na"] = res["padj"].isna()
    res["lfc_is_na"] = res["log2FoldChange"].isna()
    res["padj"] = res["padj"].fillna(1.0)
    res["log2FoldChange"] = res["log2FoldChange"].fillna(0.0)
    res["stat"] = res["stat"].fillna(0.0)
    return res.sort_values("padj")

# ═══════════════════════════════════════════
# 4. PLOTTING & EXPORT HELPERS
# ═══════════════════════════════════════════
def plot_volcano_plotly(df, padj_th, lfc_th, up_c, down_c, template='plotly_white', font="sans-serif", highlight_gene=None, font_size=12):
    df_plot = df.copy()
    df_plot['-log10(padj)'] = -np.log10(df_plot['padj'].replace(0, 1e-300))
    df_plot['Status'] = 'NS'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] > lfc_th), 'Status'] = 'Up'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] < -lfc_th), 'Status'] = 'Down'
    df_plot = df_plot.reset_index()
    gene_col = df_plot.columns[0]
    fig = px.scatter(df_plot, x='log2FoldChange', y='-log10(padj)', color='Status', 
                     hover_name=gene_col, color_discrete_map={'Up': up_c, 'Down': down_c, 'NS': '#7f8c8d'}, 
                     template=template, title="Volcano Plot")
    fig.add_hline(y=-np.log10(padj_th), line_dash="dash", line_color="gray")
    fig.add_vline(x=lfc_th, line_dash="dash", line_color="gray")
    fig.add_vline(x=-lfc_th, line_dash="dash", line_color="gray")

    if highlight_gene and highlight_gene in df_plot[gene_col].values:
        h_df = df_plot[df_plot[gene_col] == highlight_gene]
        fig.add_trace(go.Scatter(
            x=h_df['log2FoldChange'], y=h_df['-log10(padj)'],
            mode='markers+text',
            marker=dict(color='yellow', size=15, symbol='star', line=dict(width=2, color='black')),
            text=[highlight_gene], textposition="top center",
            name="Highlighted"
        ))

    fig.update_layout(font=dict(family=font, size=font_size))
    return fig

def plot_ma_plotly(df, padj_th, lfc_th=1.0, up_c="#E64B35", down_c="#4DBBD5", template='plotly_white', font="sans-serif", highlight_gene=None, font_size=12):
    df_plot = df.copy().reset_index()
    gene_col = df_plot.columns[0]
    df_plot['Status'] = 'NS'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] >  lfc_th), 'Status'] = 'Up'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] < -lfc_th), 'Status'] = 'Down'
    fig = px.scatter(df_plot, x='baseMean', y='log2FoldChange', color='Status', 
                     hover_name=gene_col, log_x=True,
                     color_discrete_map={'Up': up_c, 'Down': down_c, 'NS': '#7f8c8d'}, 
                     template=template, title="MA Plot")
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    if highlight_gene and highlight_gene in df_plot[gene_col].values:
        h_df = df_plot[df_plot[gene_col] == highlight_gene]
        fig.add_trace(go.Scatter(
            x=h_df['baseMean'], y=h_df['log2FoldChange'],
            mode='markers+text',
            marker=dict(color='yellow', size=15, symbol='star', line=dict(width=2, color='black')),
            text=[highlight_gene], textposition="top center",
            name="Highlighted"
        ))

    fig.update_layout(font=dict(family=font, size=font_size))
    return fig

def plot_pca_plotly(pca_df, explained_var, cond_colors, template='plotly_white', font="sans-serif", font_size=12):
    df_plot = pca_df.reset_index()
    sample_col = df_plot.columns[0]
    fig = px.scatter(df_plot, x='PC1', y='PC2', color='condition', 
                     hover_name=sample_col, text=sample_col, 
                     color_discrete_map=cond_colors, 
                     labels={'PC1': f'PC1 ({explained_var[0]:.1%})', 'PC2': f'PC2 ({explained_var[1]:.1%})'}, 
                     template=template, title="PCA Plot")
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')), textposition='top center')
    fig.update_layout(font=dict(family=font, size=font_size))
    return fig

def plot_corr_heatmap_plotly(df, template='plotly_white', font="sans-serif", font_size=12):
    corr = df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', 
                    template=template, title="Sample Correlation Heatmap")
    fig.update_layout(font=dict(family=font, size=font_size))
    return fig

def plot_enrich_dot_plotly(df, title, template='plotly_white', font="sans-serif", font_size=12):
    """Plotly Dot Plot for Enrichr (KEGG/GO) results."""
    df_plot = df.copy()
    try:
        df_plot['Count'] = df_plot['Overlap'].apply(lambda x: int(x.split('/')[0]) if isinstance(x, str) else 0)
    except (AttributeError, TypeError, ValueError):
        df_plot['Count'] = 10
    
    fig = px.scatter(df_plot, x='Combined Score', y='Term', size='Count', color='Adjusted P-value',
                     hover_data=['Adjusted P-value', 'Overlap'],
                     title=title, template=template, color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"))
    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=font, size=font_size))
    return fig

def plot_gsea_bar_plotly(df, title, template='plotly_white', font="sans-serif", font_size=12):
    """Plotly Bar Plot for GSEA results (NES)."""
    fig = px.bar(df, x='NES', y='Term', orientation='h', title=title,
                 color='FDR q-val', color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"), template=template)
    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=font, size=font_size))
    return fig

def plot_gsea_dot_plotly(df, title, template='plotly_white', font="sans-serif", font_size=12):
    """Plotly Dot Plot for GSEA results (NES)."""
    df_plot = df.copy()
    try:
        if 'Tag %' in df_plot.columns:
            df_plot['Size'] = df_plot['Tag %'].apply(lambda x: float(x.replace('%','')) if isinstance(x, str) else x)
        else:
            df_plot['Size'] = 10
    except (AttributeError, TypeError, ValueError):
        df_plot['Size'] = 10
        
    fig = px.scatter(df_plot, x='NES', y='Term', size='Size', color='FDR q-val',
                     hover_data=['FDR q-val', 'NOM p-val'],
                     title=title, template=template, color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"))
    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=font, size=font_size))
    return fig

def collect_all_results():
    """Collect existing result files plus one shared provenance document."""
    files = {}
    if st.session_state["counts_df"] is not None:
        files["0.1_Raw_Counts.csv"] = st.session_state["counts_df"].to_csv()
    if st.session_state.get("qc_filtered_df") is not None:
        files["0.2_QC_Filtered_Counts.csv"] = st.session_state["qc_filtered_df"].to_csv()
    if st.session_state["deg_results"] is not None:
        files["1.1_DEG_Results.csv"] = st.session_state["deg_results"].to_csv()
    if st.session_state.get("tf_collectri") is not None:
        files["2.1_TF_CollecTRI.csv"] = st.session_state["tf_collectri"].to_csv()
    if st.session_state.get("tf_dorothea") is not None:
        files["2.2_TF_DoRothEA.csv"] = st.session_state["tf_dorothea"].to_csv()
    if st.session_state.get("ciber_results") is not None:
        files["3.1_Immune_Deconvolution.csv"] = st.session_state["ciber_results"].to_csv()
        
    if st.session_state["analysis_log"]:
        nb_md = "# Analysis Notebook\n\n"
        for entry in st.session_state["analysis_log"]:
            nb_md += f"### [{entry['time']}] {entry['action']}\n{entry['details']}\n\n"
        files["Analysis_Notebook.md"] = nb_md
    if st.session_state["deg_results"] is not None:
        matrix, rna_counts = brim_provenance.describe_rna_data(
            files["0.1_Raw_Counts.csv"], st.session_state["deg_results"],
            st.session_state.get("metadata"),
        )
        rna_counts["qc_genes"] = (
            len(st.session_state["qc_filtered_df"])
            if st.session_state.get("qc_filtered_df") is not None else None
        )
        events = st.session_state["external_service_events"]
        manifest = brim_provenance.build_manifest(
            inputs={"rna": {"source_mode": "count_matrix", "count_matrix": matrix,
                            "source_files": st.session_state["rna_input_files"],
                            "is_sample_data": st.session_state.get("is_sample_data", False)}},
            settings={
                "app_version": APP_VERSION,
                "species": st.session_state.get("sp", {}).get("org", "unknown"),
                "genome_build": None,
                "rna": {
                    "lfc_threshold": st.session_state.get("lfc_t", 1.0),
                    "padj_threshold": st.session_state.get("padj_t", 0.05),
                    "normalization": st.session_state.get("norm_method", "log1p"),
                    "low_count_filtering": {
                        "enabled": st.session_state.get("filter_enable", False),
                        "min_count": st.session_state.get("filter_min_count", 10),
                        "min_samples": st.session_state.get("filter_min_samples", 2),
                    },
                    "contrast": st.session_state.get("last_contrast", ""),
                    "analysis_log": st.session_state.get("analysis_log", []),
                    "gene_id_mapping": st.session_state["rna_id_mapping"],
                    "log2fc_inverted": False,
                },
            },
            counts={"rna": rna_counts},
            services={"external_services_used": sorted({event["service"] for event in events
                                                       if event.get("lookup_outcome") in ("success", "partial")}),
                      "events": events,
                      "external_service_events": st.session_state["external_service_history"]},
        )
        files["Provenance/manifest.json"] = json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)
        files["Provenance/manifest.md"] = brim_provenance.render_manifest_markdown(manifest)
    return files

# — Variable Initialization (Global UI/Plot variables)
_language_display_options = list(LANGUAGE_OPTIONS.keys())
if st.session_state.get("lang_display") not in _language_display_options:
    st.session_state["lang_display"] = "日本語"
if st.session_state.get("language_selector") not in _language_display_options:
    st.session_state["language_selector"] = st.session_state["lang_display"]


def _sync_language_selection():
    selected = st.session_state.get("language_selector")
    if selected in LANGUAGE_OPTIONS:
        st.session_state["lang_display"] = selected


_is_jp      = st.session_state.get("lang_display", "日本語") == "日本語"
lang        = LANGUAGE_OPTIONS[st.session_state.get("lang_display", "日本語")]
lfc_t       = st.session_state.get("lfc_t", 1.0)
padj_t      = st.session_state.get("padj_t", 0.05)
up_color    = st.session_state.get("up_color", "#E64B35")
down_color  = st.session_state.get("down_color", "#4DBBD5")
fig_width   = st.session_state.get("fig_width", 800)
fig_height  = st.session_state.get("fig_height", 500)
fig_font_sz = st.session_state.get("fig_font_sz", 12)
img_format  = st.session_state.get("img_format", "png")
img_dpi     = st.session_state.get("img_dpi", 300)
sel_font    = st.session_state.get("selected_font", "sans-serif")

# — Sidebar: Analysis Status (Quick Navigation) ───
with st.sidebar:
    st.markdown(f"### {ui('📊 Analysis Status', lang, '📊 解析ステータス')}")

    # ── 各ステップの完了判定 ──────────────────────────────
    _upload_mode_sb = st.session_state.get("upload_mode")
    if _upload_mode_sb is None: _upload_mode_sb = "single" # Default to single for status display logic
    s1 = "✅" if st.session_state.get("counts_df") is not None else "⏳"
    s2 = "✅" if st.session_state.get("metadata") is not None else "⏳"
    s3 = "✅" if st.session_state.get("deg_results") is not None else "⏳"
    s4 = "✅" if st.session_state.get("tf_collectri") is not None else "⏳"

    # Multi Study専用ステータス
    _n_batch_sb   = len(st.session_state.get("batch_deg_results", {}))
    _n_studies_sb = len(st.session_state.get("multi_study_names", []))
    s_batch = "✅" if _n_batch_sb >= 2 else ("⏳" if _n_batch_sb == 1 else "⏳")
    s_meta  = "✅" if st.session_state.get("lfc_meta_matrix") is not None else "⏳"

    # ── モード別表示 ─────────────────────────────────────
    if _upload_mode_sb == "single":
        _status_items = [
            (s1, ui('Upload Data', lang, 'データアップロード')),
            (s2, ui('Group Assignment', lang, '群の設定')),
            (s3, ui('DEG Analysis', lang, 'DEG解析')),
            (s4, ui('TF / Network', lang, 'TF / Network解析')),
        ]
        # 次のアクション案内
        if st.session_state.get("counts_df") is None:
            _next = "⬆️ " + (ui('Upload your data in the Upload tab', lang, 'Uploadタブでデータをアップロード'))
        elif st.session_state.get("metadata") is None:
            _next = "⬆️ " + (ui('Assign groups in the Upload tab', lang, 'Uploadタブで群を設定してください'))
        elif st.session_state.get("deg_results") is None:
            _next = "➡️ " + (ui('Run Analyze in the DEG tab', lang, 'DEGタブでAnalyzeを実行'))
        else:
            _next = "➡️ " + (ui('Go to Visualization or Network tab', lang, 'VisualizationまたはNetworkタブへ'))
    else:
        _status_items = [
            (s1, ui('Upload ({value_0} Studies)', lang, 'データアップロード（{value_0} Study）', value_0=_n_studies_sb)),
            (s2, ui('Group / Study Assignment', lang, '群・Study設定')),
            (s_batch, ui('Batch DEG ({value_0} contrasts)', lang, '一括DEG解析（{value_0}コントラスト）', value_0=_n_batch_sb)),
            (s_meta, ui('Meta LFC Integration', lang, 'Meta LFC統合')),
            (s4, ui('TF / Network', lang, 'TF / Network解析')),
        ]
        if st.session_state.get("counts_df") is None:
            _next = "⬆️ " + (ui('Upload files in the Upload tab', lang, 'Uploadタブでファイルをアップロード'))
        elif _n_batch_sb < 2:
            _next = "➡️ " + (ui('Run Batch DEG (2+ contrasts) in DEG tab', lang, 'DEGタブで一括実行（2コントラスト以上）'))
        elif st.session_state.get("lfc_meta_matrix") is None:
            _next = "➡️ " + (ui('Run LFC integration in Meta tab', lang, 'MetaタブでLFC統合を実行'))
        else:
            _next = "✅ " + (ui('All steps complete', lang, '全ステップ完了'))

    # ── ステータスカード描画 ─────────────────────────────
    _items_html = "".join([
        f"<div style='font-size:16px; margin-bottom:5px;'>{_s} <b>{_label}</b></div>"
        for _s, _label in _status_items
    ])
    st.markdown(f"""
    <div style='background:rgba(79,110,247,0.05); padding:10px; border-radius:8px; border:1px solid rgba(79,110,247,0.1); margin-bottom:10px;'>
        {_items_html}
    </div>
    <div style='background:rgba(79,110,247,0.08); padding:8px 10px; border-radius:6px; border-left:3px solid #4F6EF7; font-size:16px; color:#4F6EF7; margin-bottom:15px;'>
        {_next}
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# — Sidebar: Language & Theme
with st.sidebar.expander(ui("Language & Theme", lang), expanded=True):
    _lang_opts = list(LANGUAGE_OPTIONS.keys())
    sel_lang_display = st.selectbox(
        ui("Language", lang), _lang_opts, label_visibility="collapsed",
        key="language_selector",
        on_change=_sync_language_selection,
    )
    st.session_state["lang_display"] = sel_lang_display
    lang = LANGUAGE_OPTIONS[sel_lang_display]

    _theme_ids = ["light", "dark", "ocean"]
    _theme_labels = [t("theme_light", lang), t("theme_dark", lang), "Ocean"]
    _saved_id = st.session_state.get("theme_id")
    if _saved_id not in _theme_ids:
        _legacy_theme = st.session_state.get("theme_choice", "")
        _saved_id = "ocean" if "Ocean" in _legacy_theme else (
            "dark" if ("Dark" in _legacy_theme or "ダーク" in _legacy_theme) else "light"
        )
    sel_theme = st.selectbox(
        ui("Theme", lang), _theme_labels, index=_theme_ids.index(_saved_id), key="theme_selector"
    )
    st.session_state["theme_id"] = _theme_ids[_theme_labels.index(sel_theme)]
    st.session_state["theme_choice"] = sel_theme

lang = LANGUAGE_OPTIONS[st.session_state.get("lang_display", "日本語")]
sel_theme = st.session_state.get("theme_choice", "Light") # BUG FIX: バグ⑤
theme_id = st.session_state.get("theme_id", "light")
is_dark  = theme_id == "dark"
is_ocean = theme_id == "ocean"
plotly_template = "plotly_dark" if (is_dark or is_ocean) else "plotly_white"

# ─── Dynamic Font Setup ───
_lang_str = st.session_state.get("lang_display", "日本語")
font_map = {
    "日本語": ("Noto Sans JP", "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&display=swap"),
    "簡体": ("Noto Sans SC", "https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;700&display=swap"),
    "繁体": ("Noto Sans TC", "https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&display=swap"),
}
# デフォルト（英語・スペイン語等）は Inter を適用
app_font_name = "Inter"
font_url = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap"

for k, (f_name, f_url) in font_map.items():
    if k in _lang_str:
        app_font_name = f_name
        font_url = f_url
        break

st.markdown(f"""
    <style>
    @import url('{font_url}');
    
    /* 基本テキストへの適用 */
    .stApp, p, h1, h2, h3, h4, h5, h6, label, li, [data-baseweb="select"], .stMarkdown {{
        font-family: '{app_font_name}', sans-serif !important;
    }}
    
    /* アイコンフォントの保護（文字化け対策） */
    [data-testid*="Icon"], .material-icons, .material-symbols-rounded {{
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Thresholds are still in sidebar as they trigger re-analysis
if st.session_state.get("upload_mode") is not None:
    with st.sidebar.expander(ui("Thresholds", lang), expanded=True):
        if "lfc_t"  not in st.session_state: st.session_state["lfc_t"]  = 1.0
        if "padj_t" not in st.session_state: st.session_state["padj_t"] = 0.05
        lfc_t  = st.slider(t("logfc_threshold", lang), 0.0, 5.0,   st.session_state["lfc_t"],  0.25, key="sb_lfc_t")
        st.caption("LFC threshold: " + (ui('Absolute log2FoldChange threshold. 1.0 = 2-fold, 2.0 = 4-fold change. Start with 1.0 if unsure.', lang, 'log2FoldChange の絶対値の閾値です。1.0 = 2倍変動、2.0 = 4倍変動に相当します。迷ったら 1.0 から始めてください。')))
        padj_t = st.slider(t("pval_threshold",  lang), 0.001, 0.1, st.session_state["padj_t"], 0.005, key="sb_padj_t")
        st.caption("padj threshold: " + (ui('Adjusted p-value threshold. 0.05 (5%) is standard. Try 0.1 if too few DEGs are found.', lang, '多重検定補正済みp値の閾値です。0.05（5%）が一般的です。DEGが少なすぎる場合は 0.1 に緩めてみてください。')))
        st.session_state["lfc_t"]  = lfc_t
        st.session_state["padj_t"] = padj_t
        _new_deg_thresholds = (lfc_t, padj_t)
        if (
            st.session_state.get("deg_t") != _new_deg_thresholds
            and st.session_state.get("deg_results") is not None
        ):
            reset_threshold_dependent_results()
        st.session_state["deg_t"] = _new_deg_thresholds

# — Sidebar: Low-count Filtering
if st.session_state.get("upload_mode") is not None:
    with st.sidebar.expander(ui("Low-count Filtering", lang), expanded=False):
        do_filter = st.checkbox(ui("Enable filtering", lang), value=False, key="filter_enable")
        min_count = st.number_input(ui("Minimum count threshold", lang), 0, 1000, 10, key="filter_min_count")
        min_samples = st.number_input(ui("Minimum samples expressing gene", lang), 1, 100, 2, key="filter_min_samples")

# ─── Dynamic Theme Injection ───
if is_dark:
    st.markdown("""
        <style>
        /* === DARK THEME (VS Code / Catppuccin Style) === */
        .stApp, [data-testid="stAppViewContainer"] { background-color: #1E1E2E !important; color: #CDD6F4 !important; }
        section[data-testid="stSidebar"] { background-color: #252537 !important; border-right: 1px solid #3E3E5E !important; }
        section[data-testid="stSidebar"] * { color: #CDD6F4 !important; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { background-color: #252537 !important; border-bottom: 1px solid #3E3E5E !important; }
        .stTabs [data-baseweb="tab"] { color: #8A95A5 !important; background: transparent !important; }
        .stTabs [aria-selected="true"] { color: #89B4FA !important; border-bottom: 2px solid #89B4FA !important; }
        .stTabs [data-baseweb="tab-panel"] { background-color: #1E1E2E !important; }

        /* Text */
        p, h1, h2, h3, h4, h5, h6, label, .stMarkdown, .stText { color: #CDD6F4 !important; }
        .stCaption { color: #8A95A5 !important; }

        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        div[data-baseweb="select"] { background-color: #1E1E2E !important; color: #CDD6F4 !important; border-color: #3E3E5E !important; }

        /* Info / Warning / Success / Error */
        div[data-testid="stInfo"]    { background-color: #2A2F45 !important; border-left: 4px solid #89B4FA !important; }
        div[data-testid="stInfo"]    * { color: #CDD6F4 !important; }
        div[data-testid="stSuccess"] { background-color: #2A3A35 !important; border-left: 4px solid #A6E3A1 !important; }
        div[data-testid="stSuccess"] * { color: #A6E3A1 !important; }
        div[data-testid="stWarning"] { background-color: #3B332A !important; border-left: 4px solid #F9E2AF !important; }
        div[data-testid="stWarning"] * { color: #F9E2AF !important; }
        div[data-testid="stError"]   { background-color: #3D2930 !important; border-left: 4px solid #F38BA8 !important; }
        div[data-testid="stError"]   * { color: #F38BA8 !important; }

        /* Plotly chart */
        [data-testid="stPlotlyChart"] > div,
        .js-plotly-plot, .plotly, .plot-container { background-color: transparent !important; }
        canvas { background-color: transparent !important; }
        iframe { background: transparent !important; }

        /* Buttons */
        .stButton > button { background-color: #252537 !important; color: #CDD6F4 !important; border: 1px solid #3E3E5E !important; }
        .stButton > button[kind="primary"] { background-color: #89B4FA !important; color: #1E1E2E !important; border: none !important; }
        .stDownloadButton > button { background-color: #252537 !important; color: #CDD6F4 !important; border: 1px solid #3E3E5E !important; }

        /* Metrics, expanders (with shadows) */
        .stMetric { background-color: #252537 !important; border-radius: 8px; padding: 8px; border: 1px solid #3E3E5E !important; box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2) !important; }
        .stMetric label, .stMetric [data-testid="stMetricValue"] { color: #CDD6F4 !important; }
        .stExpander, div[data-testid="stExpander"] { background-color: #252537 !important; border: 1px solid #3E3E5E !important; border-radius: 8px !important; box-shadow: 0 2px 8px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2) !important; }
        div[data-testid="stExpander"] summary { color: #CDD6F4 !important; }
        hr { border-color: #3E3E5E !important; }
        .stSlider > label { color: #CDD6F4 !important; }
        </style>
    """, unsafe_allow_html=True)

elif is_ocean:
    st.markdown("""
        <style>
        /* === OCEAN THEME (Deep Sea Refined) === */
        .stApp, [data-testid="stAppViewContainer"] { background-color: #060F20 !important; color: #B8E4F0 !important; }
        section[data-testid="stSidebar"] { background-color: #0A1628 !important; border-right: 1px solid #0E4272 !important; }
        section[data-testid="stSidebar"] * { color: #B8E4F0 !important; }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { background-color: #0A1628 !important; border-bottom: 1px solid #0E4272 !important; }
        .stTabs [data-baseweb="tab"] { color: #5B849C !important; background: transparent !important; }
        .stTabs [aria-selected="true"] { color: #00B8D9 !important; border-bottom: 2px solid #00B8D9 !important; }
        .stTabs [data-baseweb="tab-panel"] { background-color: #060F20 !important; }

        /* Text */
        p, h1, h2, h3, h4, h5, h6, label, .stMarkdown { color: #B8E4F0 !important; }
        .stCaption { color: #5B849C !important; }

        /* Inputs */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stNumberInput > div > div > input,
        div[data-baseweb="select"] { background-color: #060F20 !important; color: #B8E4F0 !important; border-color: #0E4272 !important; }

        /* Info / boxes */
        div[data-testid="stInfo"]    { background-color: #0A1D38 !important; border-left: 4px solid #00B8D9 !important; }
        div[data-testid="stInfo"]    * { color: #B8E4F0 !important; }
        div[data-testid="stSuccess"] { background-color: #0A2822 !important; border-left: 4px solid #06D6A0 !important; }
        div[data-testid="stSuccess"] * { color: #06D6A0 !important; }
        div[data-testid="stWarning"] { background-color: #211A0A !important; border-left: 4px solid #FFB703 !important; }
        div[data-testid="stWarning"] * { color: #FFD166 !important; }
        div[data-testid="stError"]   { background-color: #210A0A !important; border-left: 4px solid #EF233C !important; }
        div[data-testid="stError"]   * { color: #FF8FA3 !important; }

        /* Plotly chart */
        [data-testid="stPlotlyChart"] > div,
        .js-plotly-plot, .plotly, .plot-container { background-color: transparent !important; }
        canvas { background-color: transparent !important; }

        /* Buttons */
        .stButton > button { background-color: #0A1628 !important; color: #B8E4F0 !important; border: 1px solid #0E4272 !important; }
        .stButton > button[kind="primary"] { background-color: #00B8D9 !important; color: #060F20 !important; border: none !important; }
        .stDownloadButton > button { background-color: #0A1628 !important; color: #B8E4F0 !important; border: 1px solid #0E4272 !important; }

        /* Metrics, expanders (with shadows) */
        .stMetric { background-color: #0A1628 !important; border-radius: 8px; padding: 8px; border: 1px solid #0E4272 !important; box-shadow: 0 2px 8px rgba(0,180,220,0.08), 0 1px 2px rgba(0,0,0,0.2) !important; }
        .stMetric label, .stMetric [data-testid="stMetricValue"] { color: #B8E4F0 !important; }
        .stExpander, div[data-testid="stExpander"] { background-color: #0A1628 !important; border: 1px solid #0E4272 !important; border-radius: 8px !important; box-shadow: 0 2px 8px rgba(0,180,220,0.08), 0 1px 2px rgba(0,0,0,0.2) !important; }
        div[data-testid="stExpander"] summary { color: #B8E4F0 !important; }
        hr { border-color: #0E4272 !important; }
        .stSlider > label { color: #B8E4F0 !important; }
        </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
        <style>
        /* === LIGHT THEME === */
        .stApp, [data-testid="stAppViewContainer"] { background-color: #F5F7FA !important; color: #24292F; }
        section[data-testid="stSidebar"] { background-color: #EDF0F5 !important; border-right: 1px solid #D1D9E6 !important; }
        .stTabs [data-baseweb="tab-list"] { border-bottom: 1px solid #D1D9E6 !important; }
        .stExpander, div[data-testid="stExpander"] { 
            border: 1px solid #D1D9E6 !important; 
            border-radius: 8px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04) !important; 
            background-color: #FFFFFF !important;
        }
        hr { border-color: #D1D9E6 !important; }
        [data-testid="stPlotlyChart"] > div { background-color: transparent !important; }
        </style>
    """, unsafe_allow_html=True)

if st.session_state.get("upload_mode") is not None:

    # サンプルデータ使用中のバッジ表示
    if st.session_state.get("is_sample_data", False):
        _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
        if st.session_state.get("upload_mode") == "multi":
            msg = ui('🧪 **Using Multi-Study Sample Data (Mouse, 18 samples, 3 studies) — Demo only**\n\nThis dataset is for demonstration purposes only. Expression patterns are artificially constructed and do not represent real experimental data or biological dispersion.', lang)
        else:
            msg = ui('🧪 **Using Sample Data (Mouse, 12 samples, 3 groups) — Demo only**\n\nThis dataset is for demonstration purposes only. Expression patterns are artificially constructed and do not represent real experimental data or biological dispersion.', lang, '🧪 **サンプルデータを使用中 (Mouse, 12 samples, 3 groups)**\n\nこのデータは機能デモ専用です。遺伝子発現パターンは人工的に設定されており、実際の実験データを代表するものではありません。')
        st.warning(msg)

tab_upload, tab_deg, tab_viz, tab_network, tab_meta, tab_export, tab_info = st.tabs([
    ui("Upload", lang), ui("DEG", lang), ui("Visualization", lang), ui("Network", lang), ui("🔬 Meta", lang), ui("Export", lang), ui("Info", lang)
])

# TAB 1: UPLOAD
with tab_upload:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"

    # ── Mode initialisation ──────────────────────────────────────────────
    if "upload_mode" not in st.session_state:
        st.session_state["upload_mode"] = None
    _mode = st.session_state["upload_mode"]

    # ── Mode selection cards ─────────────────────────────────────────────
    _card_col1, _card_col2 = st.columns(2)

    def _card_style(selected: bool) -> str:
        base = "background: white; border: 1px solid #E5E7EB; border-radius: 24px; padding: 48px 40px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); min-height: 320px; display: flex; flex-direction: column;"
        if selected:
            base = base.replace("border: 1px solid #E5E7EB;", "border: 2px solid #4F6EF7;")
        return base

    _single_active = _mode == "single"
    _multi_active  = _mode == "multi"

    with _card_col1:
        with st.container():
            st.markdown(f"""
<div style='{_card_style(_single_active)}'>
  <div style='font-size:64px; margin-bottom:20px; text-align:left;'>🔬</div>
  <div style='font-weight:800; font-size:32px; color:#111827; margin-bottom:12px; text-align:left;'>Single Study</div>
  <div style='font-size:18px; color:#6B7280; line-height:1.6; margin-bottom:24px; text-align:left;'>
    1つの実験データを群間比較する
  </div>
</div>
""", unsafe_allow_html=True)
            if st.button(ui("→ Start with a Single Study", lang),
                         key="mode_single", use_container_width=True,
                         type="primary" if _single_active else "secondary"):
                st.session_state["upload_mode"] = "single"
                st.rerun()

    with _card_col2:
        with st.container():
            st.markdown(f"""
<div style='{_card_style(_multi_active)}'>
  <div style='font-size:64px; margin-bottom:20px; text-align:left;'>📚</div>
  <div style='font-weight:800; font-size:32px; color:#111827; margin-bottom:12px; text-align:left;'>Multi Study</div>
  <div style='font-size:18px; color:#6B7280; line-height:1.6; margin-bottom:24px; text-align:left;'>
    複数の実験を統合して比較する
  </div>
</div>
""", unsafe_allow_html=True)
            if st.button(ui("→ Start with Multiple Studies", lang),
                         key="mode_multi", use_container_width=True,
                         type="primary" if _multi_active else "secondary"):
                st.session_state["upload_mode"] = "multi"
                st.rerun()

    if _mode is not None:
        st.write("")

    def show_validation_card(df, is_jp):
        with st.container(border=True):
            st.markdown(ui('#### ✅ Data Validation Summary', lang, '#### ✅ データバリデーション結果'))
            
            # Integrated Metrics (Genes, Samples, Zero counts)
            m1, m2, m3 = st.columns(3)
            n_rows = df.shape[0]
            n_cols = df.shape[1]
            zero_rate = (df == 0).sum().sum() / df.size if df.size > 0 else 0
            
            m1.metric(ui('Genes', lang, '遺伝子数 / Genes'), f"{n_rows:,}")
            m2.metric(ui('Samples', lang, 'サンプル数 / Samples'), f"{n_cols}")
            m3.metric(ui('Zero rate', lang, 'ゼロ率 / Zero rate'), f"{zero_rate:.1%}")
            
            st.divider()
            
            c1, c2 = st.columns(2)
            
            # 0. Negative value check（DESeq2クラッシュの原因）
            _has_negative = (df.values < 0).any()
            if _has_negative:
                _n_neg = (df.values < 0).sum()
                c1.markdown(f"- 🚨 **{ui('Negative values', lang, '負値')}**: " + (
                    ui('{value_0:,} negative values detected. DESeq2 will crash. Please check your data.', lang, '{value_0:,}件の負値が含まれています。DESeq2はクラッシュします。データを確認してください。', value_0=_n_neg)
                ))
            else:
                c1.markdown(f"- ✅ **{ui('No negative values', lang, '負値なし')}**: {ui('OK', lang, '正常')}")

            # 0b. NA/NaN check
            _has_na = df.isnull().any().any()
            if _has_na:
                _n_na = df.isnull().sum().sum()
                c1.markdown(f"- 🚨 **{ui('Missing values (NA)', lang, '欠損値')}**: " + (
                    ui('{value_0:,} NA values detected. Please remove or fill with 0 before analysis.', lang, '{value_0:,}件のNAが含まれています。解析前に除去または0埋めしてください。', value_0=_n_na)
                ))
            else:
                c1.markdown(f"- ✅ **{ui('No missing values', lang, '欠損値なし')}**: {ui('OK', lang, '正常')}")

            # 1. Integer check
            is_int = np.issubdtype(df.values.dtype, np.integer) or (df.values == df.values.astype(int)).all()
            if is_int:
                c1.markdown(f"- ✅ **{ui('Data Type', lang, '数値形式')}**: {ui('Integers', lang, 'すべて整数です')}")
            else:
                c1.markdown(f"- ❌ **{ui('Data Type', lang, '数値形式')}**: " + (ui('Decimals detected. Please check if this is count data.', lang, '小数が含まれています。カウントデータか確認してください')))
            
            # 2. Duplicate check
            dups = df.index.duplicated().sum()
            if dups == 0:
                c1.markdown(f"- ✅ **{ui('Duplicates', lang, '遺伝子名の重複')}**: {ui('None', lang, 'なし')}")
            else:
                c1.markdown(f"- ⚠️ **{ui('Duplicates', lang, '遺伝子名の重複')}**: " + (ui('Merged {value_0} duplicate genes', lang, '{value_0}件の重複遺伝子を自動統合しました', value_0=dups)))
            
            # 3. Sample name check
            invalid_cols = [c for c in df.columns if re.search(r'[^a-zA-Z0-9_.]', str(c)) or ' ' in str(c)]
            if not invalid_cols:
                c2.markdown(f"- ✅ **{ui('Sample Names', lang, 'サンプル名')}**: {ui('Clean', lang, '正常')}")
            else:
                c2.markdown(f"- ⚠️ **{ui('Sample Names', lang, 'サンプル名')}**: " + (ui('Contains spaces or symbols. Underscores recommended.', lang, 'スペースや記号が含まれています。アンダースコアへの置換を推奨します')))

    # ════════════════════════════════════════════════════════════════════
    #  SINGLE STUDY MODE
    # ════════════════════════════════════════════════════════════════════
    if _mode == "single":

        # Normalization
        _norm_label = ui("Normalization method", lang)
        _norm_opts = ["log1p", "CPM", "TPM", "VST"]
        _norm_sel = st.radio(_norm_label, _norm_opts, horizontal=True, key="norm_method")
        _previous_norm_method = st.session_state.get("analysis_norm_method")
        if _previous_norm_method is not None and _previous_norm_method != _norm_sel:
            st.session_state["tf_results"] = None
            st.session_state["tf_collectri"] = None
            st.session_state["tf_dorothea"] = None
        st.session_state["analysis_norm_method"] = _norm_sel
        _norm_desc = {
            "log1p": ui("log1p(CPM) — A simple transformation for visualization and exploratory analysis. DEG uses raw counts; VST may be preferable for PCA or correlation depending on the dataset.", lang),
            "CPM": ui("Counts Per Million — library-size correction for visualization.", lang),
            "TPM": ui("Transcripts Per Million — gene-length correction for visualization. DEG uses raw counts; interpret cross-sample composition carefully.", lang),
            "VST": ui("Variance Stabilizing Transformation — a DESeq2-like PyDESeq2 transformation that may differ from the original DESeq2 implementation.", lang),
        }
        st.caption(_norm_desc[_norm_sel])
        
        # UI for Gene Length upload when TPM is selected
        if _norm_sel == "TPM":
            st.divider()
            st.markdown("#### 📏 " + (ui('Upload Gene Lengths', lang, '遺伝子長データのアップロード')))
            st.info(ui('TPM calculation requires gene lengths. Please upload a file with Gene IDs in the 1st column and lengths (bp, etc.) in the 2nd column.', lang, 'TPM計算には各遺伝子の長さ情報が必要です。1列目にGene ID、2列目に長さ（bp等）を持つファイル形式を想定しています。'))
            uploaded_lengths = st.file_uploader(ui("Gene lengths (CSV/TSV)", lang), type=["csv", "tsv", "txt"], key="lengths_uploader")
            if uploaded_lengths:
                lname = uploaded_lengths.name.lower()
                lsep = "\t" if lname.endswith((".tsv", ".txt")) else ","
                # ファイルの1行目を確認してheaderの有無を自動判定する
                # UI言語ではなくファイルの中身で判定する
                _raw_first = pd.read_csv(uploaded_lengths, sep=lsep, nrows=1, header=None)
                uploaded_lengths.seek(0)  # ファイルポインタをリセット
                _first_val = _raw_first.iloc[0, 1] if _raw_first.shape[1] >= 2 else None
                try:
                    float(_first_val)
                    # 1行目の2列目が数値 → headerなし
                    len_df = pd.read_csv(uploaded_lengths, sep=lsep, index_col=0, header=None)
                except (ValueError, TypeError):
                    # 1行目の2列目が文字列 → headerあり
                    uploaded_lengths.seek(0)
                    len_df = pd.read_csv(uploaded_lengths, sep=lsep, index_col=0, header=0)
                # If no header, assume col 1 is ID, col 2 is length
                if len(len_df.columns) >= 1:
                    try:
                        _validated_lengths = prepare_gene_lengths(len_df.iloc[:, 0])
                    except ValueError as _length_error:
                        st.session_state["gene_lengths"] = None
                        st.error(ui("Invalid gene length data: {error}", lang, error=_length_error))
                    else:
                        st.session_state["gene_lengths"] = _validated_lengths
                        st.session_state["tf_results"] = None
                        st.session_state["tf_collectri"] = None
                        st.session_state["tf_dorothea"] = None
                        st.success(ui("✅ Gene lengths loaded for {count} genes.", lang,
                                      count=len(st.session_state['gene_lengths'])))
                else:
                    st.error(ui("Invalid gene length file format.", lang))
            elif st.session_state.get("gene_lengths") is None:
                st.warning(ui("⚠️ TPM requires gene length data. Please upload a gene length file above.", lang))

        ul, ur = st.columns([1, 1], gap="large")
        with ul:
            btn_text = ui('🧪 Try with Sample Data', lang, '🧪 サンプルデータで試す')
            if st.button(btn_text, width="stretch"):
                with st.status(ui("🎩 Loading...", lang)) as status:
                    cdf, meta = generate_sample_data()
                    reset_data_results()
                    st.session_state["rna_input_files"] = []
                    st.session_state["rna_id_mapping"] = []
                    st.session_state["external_service_events"] = []
                    st.session_state["counts_df"] = cdf
                    st.session_state["qc_filtered_df"] = cdf
                    st.session_state["metadata"] = meta
                    st.session_state["conditions"] = ["Control", "Treatment_A", "Treatment_B"]
                    st.session_state["sp"] = SPECIES_MAP["Mouse (mmu)"]
                    st.session_state["is_sample_data"] = True
                    # Show validation for sample data too
                    st.session_state["last_validation_df"] = cdf
                    status.update(label=ui("✅ Sample data loaded (500 genes, 3 groups)", lang), state="complete", expanded=False)
                st.rerun()

            st.divider()
            uploaded_counts = st.file_uploader(t("count_matrix", lang), type=["csv", "tsv", "txt"])
            if uploaded_counts:
                sp_sel = st.selectbox(t("species", lang), list(SPECIES_MAP.keys()))
                _selected_species = SPECIES_MAP[sp_sel]
                id_mode_sel = st.radio(t("gene_id_mode", lang), [t("gene_symbol_opt", lang), t("gene_ids_opt", lang)])
                if id_mode_sel == t("gene_ids_opt", lang):
                    st.info(ui("Loading sends gene IDs to mygene.info for symbol mapping.", lang,
                               "読み込み時に遺伝子IDを mygene.info へ送信してsymbolに変換します。"))
                if st.button(ui("Load", lang), type="primary"):
                    with st.status(ui("🎩 Processing...", lang), expanded=True) as status, service_input_attempt() as _load_services:
                        try:
                            name = uploaded_counts.name.lower()
                            sep = "\t" if name.endswith((".tsv", ".txt")) else ","
                            _source_file = {"file_name": uploaded_counts.name,
                                            "sha256": brim_provenance.file_checksum(uploaded_counts)}
                            _load_mapping = []
                            raw_df = read_count_matrix_file(uploaded_counts, sep)
                            if id_mode_sel == t("gene_ids_opt", lang):
                                ids = [re.sub(r'\.\d+$', '', str(idx)) for idx in raw_df.index]
                                _load_services.append(external_service_record("mygene.info", "gene IDs", _source_file))
                                m = run_online_mapping(ids, "mouse" if _selected_species["org"]=="mmu" else "human", event=_load_services[-1])
                                _load_mapping.append({
                                    "file_name": uploaded_counts.name, "method": "mygene.info",
                                    "transform": "strip trailing version suffix; map IDs to symbols; retain unmatched IDs",
                                    "unique_ids": len(set(ids)), "matched_ids": len(m),
                                    "success_rate": len(m) / len(set(ids)) if ids else None,
                                })
                                if len(m) < len(set(ids)):
                                    st.warning(ui("Gene ID mapping matched {mapped} of {total} unique IDs. Unmapped IDs were retained unchanged.", lang,
                                                  mapped=len(m), total=len(set(ids))))
                                raw_df.index = [m.get(i, i) for i in ids]
                            counts_df = prepare_count_matrix(raw_df)
                        except (ValueError, TypeError, pd.errors.ParserError) as _input_error:
                            status.update(label=ui("❌ Invalid count matrix", lang), state="error", expanded=True)
                            st.error(ui("Invalid count matrix: {error}", lang, error=_input_error))
                        else:
                            for _event in _load_services:
                                _event["input_outcome"] = "accepted"
                            reset_data_results()
                            st.session_state["rna_input_files"] = [_source_file]
                            st.session_state["rna_id_mapping"] = _load_mapping
                            st.session_state["external_service_events"] = _load_services
                            st.session_state["counts_df"] = counts_df
                            st.session_state["qc_filtered_df"] = counts_df
                            st.session_state["sp"] = _selected_species
                            st.session_state["metadata"] = None
                            st.session_state["is_sample_data"] = False
                            st.session_state["last_validation_df"] = counts_df
                            status.update(label="✅ Ready!", state="complete", expanded=False)
                            st.rerun()

        with ur:
            if st.session_state.get("last_validation_df") is not None:
                show_validation_card(st.session_state["last_validation_df"], _is_jp)
                if st.button(ui('Close Validation', lang, 'バリデーションを閉じる')):
                    del st.session_state["last_validation_df"]
                    st.rerun()

            if st.session_state["counts_df"] is not None:
                df = st.session_state["counts_df"]
                # Show standalone metrics only if validation card is closed to avoid duplication
                if st.session_state.get("last_validation_df") is None:
                    c1, c2, c3 = st.columns(3)
                    c1.metric(t("genes", lang), f"{df.shape[0]:,}")
                    c2.metric(t("samples", lang), f"{df.shape[1]:,}")
                    c3.metric(t("qc_zeros", lang), f"{(df == 0).sum().sum() / df.size:.1%}")
                st.dataframe(df.head(10), width="stretch")
                ng = st.number_input(t("n_groups", lang), 2, 6, 2)
                gnames = [
                    st.text_input(ui("Group {number}", lang, number=i+1), f"G{i+1}", key=f"gn_{i}").strip()
                    for i in range(ng)
                ]
                if any(not group for group in gnames):
                    st.session_state["metadata"] = None
                    st.session_state["conditions"] = []
                    st.error(ui("Group names must not be empty.", lang))
                    st.stop()
                if len(set(gnames)) != len(gnames):
                    st.session_state["metadata"] = None
                    st.session_state["conditions"] = []
                    st.error(ui("Group names must be unique.", lang))
                    st.stop()
                sample_map = {s: st.selectbox(ui("Assign {sample}", lang, sample=s), gnames, key=f"gs_{s}") for s in df.columns}
                _new_metadata = pd.DataFrame.from_dict(
                    sample_map, orient="index", columns=["condition"]
                )
                _old_metadata = st.session_state.get("metadata")
                _condition_assignment_changed = (
                    _old_metadata is not None
                    and "condition" in _old_metadata.columns
                    and not _old_metadata["condition"].reindex(_new_metadata.index).equals(_new_metadata["condition"])
                )
                if _condition_assignment_changed:
                    _had_assignment_results = has_data_results()
                    reset_data_results()
                    if _had_assignment_results:
                        st.warning(ui("Analysis results were cleared because group assignments changed.", lang))
                if _old_metadata is not None:
                    for _metadata_column in _old_metadata.columns:
                        if _metadata_column != "condition":
                            _new_metadata[_metadata_column] = _old_metadata[_metadata_column].reindex(_new_metadata.index)
                st.session_state["metadata"] = _new_metadata
                st.session_state["conditions"] = list(dict.fromkeys(gnames))

                st.divider()
                with st.expander(
                    ui('➕ Additional variables (optional, for interaction analysis)', lang, '➕ 追加変数の設定（交互作用解析用・任意）'),
                    expanded=False
                ):
                    if _is_jp:
                        st.markdown("""
**何ができるか：**
遺伝子発現に影響する「条件以外の要因」をメタデータに追加できます。
この変数を使うと、DEGタブの **交互作用解析** で「treatmentの効果がageによってどう変わるか」を解析できます。

**具体例：**
- `age`（週齢・日齢） → 若いマウスと老齢マウスで薬の効き方が違う遺伝子を検出
- `batch`（実験バッチ番号） → バッチ間のばらつきを交絡因子として統制
- `weight`（体重） → 体重依存的な発現変動を解析

**設定方法：**
1. 追加する変数の数を選択（最大3つ）
2. 変数名を入力（例: `age`）
3. 各サンプルの値を入力（数値 or 文字列）
""")
                    else:
                        st.markdown(ui("""
**What you can do:**
Add covariates beyond 'condition' to your metadata.
These variables enable **Interaction Analysis** in the DEG tab — e.g., detecting genes whose treatment response differs by age.

**Examples:**
- `age` (weeks/days) → genes whose drug response differs between young and old mice
- `batch` (experiment batch) → control for batch effects as a covariate
- `weight` (body weight) → detect weight-dependent expression changes

**How to set up:**
1. Choose the number of variables to add (max 3)
2. Enter a variable name (e.g., `age`)
3. Enter each sample's value (numeric or string)
""", lang))
                    _n_extra_vars = st.number_input(
                        ui('Number of additional variables', lang, '追加変数の数'),
                        min_value=0, max_value=3, value=0, step=1, key="n_extra_vars"
                    )
                    if _n_extra_vars > 0:
                        _extra_var_names = []
                        for _vi in range(int(_n_extra_vars)):
                            _vname = st.text_input(
                                ui('Variable name {value_0}', lang, '変数名 {value_0}', value_0=_vi + 1),
                                f"var{_vi+1}",
                                key=f"extra_var_name_{_vi}"
                            ).strip()
                            _extra_var_names.append(_vname)
                        _reserved_variable_names = {"condition", "batch", "sample_label"}
                        _invalid_variable_names = [
                            name for name in _extra_var_names
                            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or "")
                            or name.lower() in _reserved_variable_names
                        ]
                        if len(set(_extra_var_names)) != len(_extra_var_names):
                            st.error(ui("Additional variable names must be unique.", lang))
                            st.stop()
                        if _invalid_variable_names:
                            st.error(ui("Additional variable names must start with a letter or underscore, contain only letters, numbers, and underscores, and must not use reserved names such as condition or batch.", lang))
                            st.stop()
                        if any(_extra_var_names):
                            st.markdown(
                                ui('**Enter values for each sample:**', lang, '**各サンプルの値を入力してください：**')
                            )
                            _extra_cols_data = {}
                            for _vname in _extra_var_names:
                                if not _vname:
                                    continue
                                _ncols = min(4, len(df.columns))
                                _vcols = st.columns(_ncols)
                                _var_vals = {}
                                for _si, _sname in enumerate(df.columns):
                                    with _vcols[_si % _ncols]:
                                        _var_vals[_sname] = st.text_input(
                                            f"{_sname}", "0",
                                            key=f"extra_var_{_vname}_{_sname}"
                                        )
                                _extra_cols_data[_vname] = _var_vals
                            _meta_updated = st.session_state["metadata"].copy()
                            for _vname, _vals in _extra_cols_data.items():
                                _meta_updated[_vname] = _meta_updated.index.map(_vals)
                                try:
                                    _meta_updated[_vname] = _meta_updated[_vname].astype(float)
                                except (ValueError, TypeError):
                                    pass
                            _interaction_columns_changed = not _meta_updated.equals(st.session_state["metadata"])
                            if _interaction_columns_changed:
                                st.session_state["ia_results"] = None
                                st.session_state["ia_all_term_results"] = {}
                                st.session_state["ia_coef_names"] = []
                                st.session_state["ia_lasso_genes"] = []
                            st.session_state["metadata"] = _meta_updated
                            st.dataframe(st.session_state["metadata"], width="stretch")
                            st.success(
                                ui('✅ {value_0} additional variable(s) configured.', lang, '✅ {value_0}個の追加変数が設定されました。', value_0=len(_extra_var_names))
                            )

                # Low-count filtering
                counts_to_use = df
                genes_removed = 0
                _filter_enabled = st.session_state.get("filter_enable", False)
                if _filter_enabled:
                    thresh = st.session_state["filter_min_count"]
                    m_samples = st.session_state["filter_min_samples"]
                    mask = (df >= thresh).sum(axis=1) >= m_samples
                    counts_to_use = df.loc[mask]
                    genes_removed = df.shape[0] - counts_to_use.shape[0]
                    if st.session_state.get("last_filter_params") != (thresh, m_samples, counts_to_use.shape[0]):
                        st.session_state["last_filter_params"] = (thresh, m_samples, counts_to_use.shape[0])
                        log_analysis("Low-count filtering applied", f"Threshold: {thresh}, Min samples: {m_samples}, Genes removed: {genes_removed}")

                _filter_signature = (
                    bool(_filter_enabled),
                    int(st.session_state.get("filter_min_count", 0)) if _filter_enabled else None,
                    int(st.session_state.get("filter_min_samples", 0)) if _filter_enabled else None,
                    int(counts_to_use.shape[0]),
                )
                _previous_filter_signature = st.session_state.get("active_filter_signature")
                if _previous_filter_signature is not None and _previous_filter_signature != _filter_signature:
                    _had_filter_results = has_data_results()
                    reset_data_results()
                    if _had_filter_results:
                        st.warning(ui("Analysis results were cleared because the filtering settings changed.", lang))
                st.session_state["active_filter_signature"] = _filter_signature
                st.session_state["qc_filtered_df"] = counts_to_use

                if st.session_state.get("filter_enable", False):
                    st.info(ui("Filtering: {before:,} → {after:,} genes ({removed:,} removed)", lang,
                               before=df.shape[0], after=counts_to_use.shape[0], removed=genes_removed))

                if counts_to_use.empty:
                    st.error(ui("No genes remain after filtering. Relax the filtering settings before analysis.", lang))
                    st.stop()

                _zero_library_qc = counts_to_use.columns[counts_to_use.sum(axis=0) <= 0].astype(str).tolist()
                if _zero_library_qc:
                    st.error(ui("Samples with zero total counts cannot be analyzed: {samples}", lang,
                                samples=", ".join(_zero_library_qc)))
                    st.stop()

                if st.session_state["metadata"] is not None:
                    st.success("✅ " + (ui("Data ready! Please go to the 'DEG' tab to run analysis.", lang, 'データ準備完了！上の『DEG』タブに移動して解析を実行してください。')))

                # QC Dashboard
                st.divider()
                st.subheader(ui("📊 QC Dashboard", lang))
                qc_df = st.session_state["qc_filtered_df"]
                _cur_norm = st.session_state.get("norm_method", "log1p")
                if _cur_norm not in ["log1p", "VST"]:
                    if _is_jp:
                        st.info(ui("💡 **log1p** or **VST** is recommended for PCA and correlation heatmap. Current: **{normalization}**",
                                   lang, normalization=_cur_norm))
                    else:
                        st.info(ui("💡 **log1p** or **VST** is recommended for PCA and correlation heatmap. Current: **{normalization}**",
                                   lang, normalization=_cur_norm))

                qct1, qct2 = st.tabs([ui("Library Size & Detected Genes", lang), ui("Correlation & PCA", lang)])
                with qct1:
                    lib_size = qc_df.sum(axis=0).reset_index()
                    lib_size.columns = ["Sample", "Total Reads"]
                    log_scale = st.checkbox(ui("Log scale (Library Size)", lang), value=False)
                    fig_lib = px.bar(lib_size, x="Sample", y="Total Reads", title="Library Size per Sample",
                                     template=plotly_template, color="Sample")
                    if log_scale:
                        fig_lib.update_layout(yaxis_type="log")
                    fig_lib.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_lib, width="stretch")
                    det_genes = (qc_df > 0).sum(axis=0).reset_index()
                    det_genes.columns = ["Sample", "Detected Genes"]
                    fig_det = px.bar(det_genes, x="Sample", y="Detected Genes", title="Detected Genes per Sample",
                                     template=plotly_template, color="Sample")
                    fig_det.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_det, width="stretch")

                with qct2:
                    _lib_size_qc = qc_df.sum(axis=0)
                    _cpm_qc = qc_df.div(_lib_size_qc, axis=1) * 1e6
                    log_qc = np.log1p(_cpm_qc)
                    st.caption(
                        ui('ℹ️ Correlation heatmap and PCA in this QC tab are always computed using **log1p(CPM)**, regardless of the normalization setting in the sidebar. The sidebar normalization (e.g. VST) applies to DEG analysis and the Visualization tab.', lang, 'ℹ️ このQCタブの相関ヒートマップとPCAは、常に **log1p(CPM)** で計算されます。サイドバーの正規化設定（VST等）はDEG解析とVisualizationタブに適用されます。')
                    )
                    corr_mat = log_qc.corr()
                    fig_corr = px.imshow(
                        corr_mat,
                        text_auto=".2f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        color_continuous_midpoint=0,
                        zmin=-1,
                        zmax=1,
                        title="Sample Correlation Heatmap (log1p-CPM)",
                        template=plotly_template
                    )
                    fig_corr.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_corr, width="stretch")

                    # Top HVG（高分散遺伝子）でPCAを計算（DESeq2標準仕様に準拠）
                    _n_hvg = min(500, log_qc.shape[0])
                    _gene_var = log_qc.var(axis=1)
                    _hvg_idx = _gene_var.nlargest(_n_hvg).index
                    log_qc_hvg = log_qc.loc[_hvg_idx]
                    if min(log_qc_hvg.shape) < 2:
                        st.info(ui("PCA requires at least 2 samples and 2 genes. The PCA plot was skipped.", lang))
                    else:
                        pca_qc = PCA(n_components=2)
                        coords_qc = pca_qc.fit_transform(log_qc_hvg.T)
                        pca_df_qc = pd.DataFrame(coords_qc, columns=["PC1", "PC2"], index=qc_df.columns)
                        pca_df_qc["condition"] = st.session_state["metadata"]["condition"]
                        fig_pca_qc = px.scatter(pca_df_qc, x="PC1", y="PC2", color="condition",
                                                text=pca_df_qc.index, title=f"Sample PCA (Top {_n_hvg} HVG)",
                                                template=plotly_template)
                        fig_pca_qc.update_traces(textposition='top center')
                        exp_var = pca_qc.explained_variance_ratio_ * 100
                        fig_pca_qc.update_layout(
                            xaxis_title=f"PC1 ({exp_var[0]:.1f}%)",
                            yaxis_title=f"PC2 ({exp_var[1]:.1f}%)",
                            font=dict(family=sel_font, size=fig_font_sz)
                        )
                        st.plotly_chart(fig_pca_qc, width="stretch")
            else:
                st.info(t("upload_prompt", lang))

    # ════════════════════════════════════════════════════════════════════
    #  MULTI STUDY MODE
    # ════════════════════════════════════════════════════════════════════
    elif _mode == "multi":

        # ── Multi Study サンプルデータボタン ────────────────────────────
        _ms_btn_text = ui('🧪 Try Sample Data (Atopic / Psoriasis / AEW)', lang, '🧪 サンプルデータで試す（Atopic / Psoriasis / AEW）')
        if st.button(_ms_btn_text, width="stretch", key="multi_sample_btn"):
            with st.status(ui("🎩 Loading...", lang), expanded=True) as _msts:
                _msdata = generate_multi_study_sample_data()
                _merged_counts_list = []
                _merged_meta_list   = []
                _loaded_study_names = []
                for _sname2, _sd in _msdata.items():
                    _study_counts = prepare_count_matrix(_sd["counts"])
                    _study_meta = _sd["metadata"].copy()
                    _sample_rename = {
                        _sample: f"[{_sname2}] {_sample}" for _sample in _study_counts.columns
                    }
                    _study_counts = _study_counts.rename(columns=_sample_rename)
                    _study_meta.index = [_sample_rename[_sample] for _sample in _study_meta.index]
                    _study_meta["condition"] = [
                        f"[{_sname2}] {_group}" for _group in _study_meta["condition"]
                    ]
                    _study_meta["batch"] = _sname2
                    _merged_counts_list.append(_study_counts)
                    _merged_meta_list.append(_study_meta)
                    _loaded_study_names.append(_sname2)
                    st.write(ui("✅ {study}: {samples} samples, {genes} genes", lang,
                                study=_sname2, samples=_sd['counts'].shape[1], genes=_sd['counts'].shape[0]))
                _mc = pd.concat(_merged_counts_list, axis=1, join="outer").fillna(0).astype(int)
                _mm = pd.concat(_merged_meta_list, axis=0)
                _mconds = list(dict.fromkeys(_mm["condition"].tolist()))
                reset_data_results()
                st.session_state["rna_input_files"] = []
                st.session_state["rna_id_mapping"] = []
                st.session_state["external_service_events"] = []
                st.session_state["counts_df"]          = _mc
                st.session_state["qc_filtered_df"]     = _mc
                st.session_state["metadata"]           = _mm
                st.session_state["conditions"]         = _mconds
                st.session_state["multi_study_names"]  = _loaded_study_names
                st.session_state["sp"]                  = SPECIES_MAP["Mouse (mmu)"]
                st.session_state["upload_mode"]        = "multi"
                st.session_state["is_sample_data"]     = True
                st.session_state["last_validation_df"] = _mc
                _msts.update(label=ui("✅ 3 studies loaded (Atopic / Psoriasis / AEW)", lang), state="complete", expanded=False)
            st.rerun()

        # ── File upload ──────────────────────────────────────────────────
        _multi_files = st.file_uploader(
            ui('📁 Count matrix files (one per study)', lang, '📁 カウント行列ファイル（Study ごとに1ファイル）'),
            type=["csv", "tsv", "txt"],
            accept_multiple_files=True,
            key="multi_files"
        )

        if _multi_files:
            # --- Added: Batch settings UI ---
            if len(_multi_files) > 1:
                _bc1, _bc2 = st.columns([3, 1])
                with _bc1:
                    st.info("💡 " + (ui('Apply settings (Species, ID mode) from the first study to all others.', lang, '最初のStudyの設定（種、IDモード）を他のすべてのStudyにコピーできます。')))
                with _bc2:
                    if st.button("✨ " + (ui('Apply to All', lang, '全Studyに適用')), key="multi_batch_apply", use_container_width=True):
                        # Copy from _0 keys to _i keys
                        _base_sp = st.session_state.get("study_sp_0")
                        _base_id = st.session_state.get("study_idmode_0")
                        for _bi in range(1, len(_multi_files)):
                            st.session_state[f"study_sp_{_bi}"] = _base_sp
                            st.session_state[f"study_idmode_{_bi}"] = _base_id
                        st.rerun()

            _study_configs = []
            for _i, _f in enumerate(_multi_files):
                _default_name = _f.name.rsplit(".", 1)[0]
                with st.expander(f"📄 {_f.name}", expanded=True):
                    _sname  = st.text_input(ui('Study name', lang, 'Study 名 / Study name'),
                                            value=_default_name, key=f"study_name_{_i}")
                    _sp_sel = st.selectbox(ui('Species', lang, '種 / Species'),
                                           list(SPECIES_MAP.keys()), key=f"study_sp_{_i}")
                    _multi_id_mode_labels = {
                        "symbol": t("gene_symbol_opt", lang),
                        "ensembl": t("gene_ids_opt", lang),
                    }
                    _id_mode = st.radio(
                        ui('Gene ID mode', lang, 'Gene ID モード / Gene ID mode'),
                        ["symbol", "ensembl"], format_func=_multi_id_mode_labels.get,
                        horizontal=True, key=f"study_idmode_{_i}"
                    )
                    _ng = int(st.number_input(ui('Number of groups', lang, '群数 / Number of groups'),
                                              2, 6, 2, key=f"study_ng_{_i}"))
                    _default_gname_map = {0: "Control", 1: "Disease", 2: "Treatment_A", 3: "Treatment_B", 4: "Group_E", 5: "Group_F"}
                    _gnames = [st.text_input(ui("Group {number} name", lang, number=_j+1), _default_gname_map.get(_j, f"G{_j+1}"),
                                             key=f"study_gn_{_i}_{_j}") for _j in range(_ng)]

                    # Read columns for sample assignment
                    try:
                        _f.seek(0)
                        _fname_lower = _f.name.lower()
                        _sep_guess = "\t" if _fname_lower.endswith((".tsv", ".txt")) else ","
                        _sample_cols = read_count_sample_names(_f, _sep_guess)
                    except Exception:
                        _sample_cols = []

                    _sample_assign = {}
                    if _sample_cols:
                        st.markdown(ui('**Sample → group assignment**', lang, '**サンプル → 群割り当て / Sample → group assignment**'))
                        # 4列固定グリッドで整列
                        _n_cols = 4
                        for _row_start in range(0, len(_sample_cols), _n_cols):
                            _row_samples = _sample_cols[_row_start:_row_start + _n_cols]
                            _sa_cols = st.columns(_n_cols)
                            for _si, _sc_name in enumerate(_row_samples):
                                _sample_assign[_sc_name] = _sa_cols[_si].selectbox(
                                    _sc_name, _gnames, key=f"study_gs_{_i}_{_sc_name}"
                                )

                    _study_configs.append({
                        "file": _f, "name": _sname, "sp": SPECIES_MAP[_sp_sel],
                        "id_mode": _id_mode, "gnames": _gnames, "sample_assign": _sample_assign
                    })

            # ── Load All button ──────────────────────────────────────────
            if any(_cfg["id_mode"] == "ensembl" for _cfg in _study_configs):
                st.info(ui("Loading sends gene IDs to mygene.info for symbol mapping.", lang,
                           "読み込み時に遺伝子IDを mygene.info へ送信してsymbolに変換します。"))
            if st.button(ui("📥 Load All Studies", lang),
                         key="multi_load_btn", type="primary"):
                _study_names_for_load = [_cfg["name"].strip() for _cfg in _study_configs]
                if any(not _name for _name in _study_names_for_load):
                    st.error(ui("Study names must not be empty.", lang))
                    st.stop()
                if len(set(_study_names_for_load)) != len(_study_names_for_load):
                    st.error(ui("Study names must be unique.", lang))
                    st.stop()
                for _cfg, _clean_study_name in zip(_study_configs, _study_names_for_load):
                    _clean_group_names = [_group.strip() for _group in _cfg["gnames"]]
                    if any(not _group for _group in _clean_group_names):
                        st.error(ui("Group names must not be empty in study {study}.", lang,
                                    study=_clean_study_name))
                        st.stop()
                    if len(set(_clean_group_names)) != len(_clean_group_names):
                        st.error(ui("Group names must be unique within study {study}.", lang,
                                    study=_clean_study_name))
                        st.stop()
                    _group_name_map = dict(zip(_cfg["gnames"], _clean_group_names))
                    _cfg["name"] = _clean_study_name
                    _cfg["gnames"] = _clean_group_names
                    _cfg["sample_assign"] = {
                        _sample: _group_name_map[_group]
                        for _sample, _group in _cfg["sample_assign"].items()
                    }
                _species_orgs = {_cfg["sp"]["org"] for _cfg in _study_configs}
                if len(_species_orgs) != 1:
                    st.error(ui("All studies must use the same species for a combined analysis.", lang))
                    st.stop()

                _all_counts = []
                _all_meta   = []
                _all_conds  = []
                _loaded_names = []
                _load_sources = []
                _load_mapping = []

                with st.status(ui("🎩 Loading...", lang), expanded=True) as _sts, service_input_attempt() as _load_services:
                    for _cfg in _study_configs:
                        _f2 = _cfg["file"]
                        _load_sources.append({"file_name": _f2.name, "study": _cfg["name"],
                                              "sha256": brim_provenance.file_checksum(_f2)})
                        _f2.seek(0)
                        _fname2 = _f2.name.lower()
                        _sep2 = "\t" if _fname2.endswith((".tsv", ".txt")) else ","
                        _raw = read_count_matrix_file(_f2, _sep2)

                        if _cfg["id_mode"] == "ensembl":
                            st.write(ui("🔗 Mapping Ensembl IDs for {study}...", lang, study=_cfg['name']))
                            _ids2 = [re.sub(r'\.\d+$', '', str(_x)) for _x in _raw.index]
                            _org2 = "mouse" if _cfg["sp"]["org"] == "mmu" else "human"
                            _load_services.append(external_service_record("mygene.info", "gene IDs", _load_sources[-1]))
                            _map2 = run_online_mapping(_ids2, _org2, event=_load_services[-1])
                            _load_mapping.append({
                                "file_name": _f2.name, "study": _cfg["name"], "method": "mygene.info",
                                "transform": "strip trailing version suffix; map IDs to symbols; retain unmatched IDs",
                                "unique_ids": len(set(_ids2)), "matched_ids": len(_map2),
                                "success_rate": len(_map2) / len(set(_ids2)) if _ids2 else None,
                            })
                            if len(_map2) < len(set(_ids2)):
                                st.warning(ui("Gene ID mapping matched {mapped} of {total} unique IDs. Unmapped IDs were retained unchanged.", lang,
                                              mapped=len(_map2), total=len(set(_ids2))))
                            _raw.index = [_map2.get(_x, _x) for _x in _ids2]

                        try:
                            _cnt = prepare_count_matrix(_raw)
                        except (ValueError, TypeError) as _input_error:
                            _sts.update(label=ui("❌ Invalid count matrix", lang), state="error", expanded=True)
                            st.error(ui("Invalid count matrix in {study}: {error}", lang,
                                        study=_cfg["name"], error=_input_error))
                            st.stop()

                        _meta_rows = {}
                        _sample_rename = {}
                        for _s, _g in _cfg["sample_assign"].items():
                            if _s in _cnt.columns:
                                _internal_sample = f"[{_cfg['name']}] {_s}"
                                _internal_condition = f"[{_cfg['name']}] {_g}"
                                _sample_rename[_s] = _internal_sample
                                _meta_rows[_internal_sample] = {
                                    "condition": _internal_condition,
                                    "batch": _cfg["name"],
                                    "sample_label": _s,
                                }

                        if len(_sample_rename) != len(_cnt.columns):
                            _sts.update(label=ui("❌ Invalid count matrix", lang), state="error", expanded=True)
                            st.error(ui("Not all samples were assigned in study {study}.", lang,
                                        study=_cfg["name"]))
                            st.stop()
                        _cnt = _cnt.rename(columns=_sample_rename)

                        _meta_df = pd.DataFrame.from_dict(_meta_rows, orient="index")
                        _all_counts.append(_cnt)
                        _all_meta.append(_meta_df)
                        _all_conds.extend([f"[{_cfg['name']}] {_g}" for _g in _cfg["gnames"]])
                        _loaded_names.append(_cfg["name"])
                        st.write(ui("✅ {study}: {genes:,} genes × {samples} samples", lang,
                                    study=_cfg['name'], genes=_cnt.shape[0], samples=_cnt.shape[1]))

                    # Merge
                    _merged_counts = pd.concat(_all_counts, axis=1, join="outer").fillna(0).astype(int)
                    _merged_meta   = pd.concat(_all_meta, axis=0)
                    _merged_conds  = list(dict.fromkeys(_all_conds))

                    if _merged_counts.columns.duplicated().any() or _merged_meta.index.duplicated().any():
                        _sts.update(label=ui("❌ Invalid count matrix", lang), state="error", expanded=True)
                        st.error(ui("Duplicate sample identifiers were detected after merging studies.", lang))
                        st.stop()

                    reset_data_results()
                    for _event in _load_services:
                        _event["input_outcome"] = "accepted"
                    st.session_state["rna_input_files"] = _load_sources
                    st.session_state["rna_id_mapping"] = _load_mapping
                    st.session_state["external_service_events"] = _load_services
                    st.session_state["counts_df"]        = _merged_counts
                    st.session_state["qc_filtered_df"]   = _merged_counts
                    st.session_state["metadata"]         = _merged_meta
                    st.session_state["conditions"]       = _merged_conds
                    st.session_state["multi_study_names"] = _loaded_names
                    st.session_state["sp"]               = _study_configs[0]["sp"]
                    st.session_state["upload_mode"]      = "multi"
                    st.session_state["is_sample_data"]   = False
                    st.session_state["last_validation_df"] = _merged_counts

                    _sts.update(label=f"✅ {len(_loaded_names)} studies loaded!", state="complete", expanded=False)

                st.success(ui("✅ Loaded {count} studies", lang, count=len(_loaded_names)))
                st.rerun()

        if st.session_state.get("last_validation_df") is not None:
            show_validation_card(st.session_state["last_validation_df"], _is_jp)
            if st.button(ui('Close Validation', lang, 'バリデーションを閉じる'), key="multi_close_v"):
                del st.session_state["last_validation_df"]
                st.rerun()

        # ── Post-load summary ────────────────────────────────────────────
        if st.session_state.get("counts_df") is not None and st.session_state.get("multi_study_names"):
            _df_m = st.session_state["counts_df"]
            _meta_m = st.session_state["metadata"]
            if st.session_state.get("last_validation_df") is None:
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric(ui('Genes', lang, '遺伝子数 / Genes'),        f"{_df_m.shape[0]:,}")
                _mc2.metric(ui('Samples', lang, 'サンプル数 / Samples'),  f"{_df_m.shape[1]:,}")
                _mc3.metric(ui('Zero rate', lang, 'ゼロ率 / Zero rate'),  f"{(_df_m == 0).sum().sum() / _df_m.size:.1%}")
            st.dataframe(_df_m.head(10), width="stretch")
            st.subheader(ui('📦 Samples per Study', lang, '📦 Study ごとのサンプル数 / Samples per Study'))
            st.dataframe(
                _meta_m["batch"].value_counts().rename_axis("Study").reset_index(name="Samples"),
                width="stretch"
            )
            st.success("✅ " + (ui("Data ready! Please go to the 'DEG' tab to run analysis.", lang, 'データ準備完了！上の『DEG』タブに移動して解析を実行してください。')))

# TAB 2: DEG
with tab_deg:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["counts_df"] is None:
        if _is_jp:
            st.info("💡 **データがありません**\n\nまずは **Upload** タブでカウント行列をアップロードし、メタデータを設定してください。")
        else:
            st.info(ui("💡 **Empty State**\n\nPlease upload a count matrix and set metadata in the **Upload** tab first.", lang))
    else:
        dl, dr = st.columns([1, 2])
        with dl:
            with st.expander(ui("⚙️ Plot Settings", lang), expanded=True):
                _threshold_col1, _threshold_col2 = st.columns(2)
                _threshold_col1.metric(t("logfc_threshold", lang), f"{lfc_t:.2f}")
                _threshold_col2.metric(t("pval_threshold", lang), f"{padj_t:.3f}")
                st.caption(ui("Change DEG thresholds in the sidebar.", lang))
                up_color = st.color_picker(ui("Up color", lang), st.session_state.get("up_color", "#E64B35"), key="deg_upc_in")
                down_color = st.color_picker(ui("Down color", lang), st.session_state.get("down_color", "#4DBBD5"), key="deg_dnc_in")
                st.session_state["up_color"] = up_color
                st.session_state["down_color"] = down_color

            with st.expander(ui("About DEG analysis", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
**DESeq2** は負の二項分布モデルに基づく差次発現解析の標準的な手法です。正規化・分散推定・仮説検定を一貫して行います。

- **log2FoldChange**: Test群 / Reference群の発現比（log2スケール）
- **padj**: Benjamini-Hochberg法による多重検定補正済みp値
- 各グループに最低2サンプル必要です（3サンプル以上を推奨）
- Cook's distance によるアウトライアー除去を自動実施
- DEG解析では、選択した2群のサンプルだけを用いて分散を推定し、他のStudyや条件はモデルへ含めません。
""")
                else:
                    st.markdown(ui("""
**DESeq2** is the standard method for differential expression analysis based on a negative binomial distribution model. It performs normalization, dispersion estimation, and hypothesis testing in a unified framework.

- **log2FoldChange**: expression ratio of Test / Reference group (log2 scale)
- **padj**: p-value corrected for multiple testing using the Benjamini-Hochberg method
- At least 2 samples per group are required (3 or more recommended)
- Outlier removal via Cook's distance is applied automatically
- DEG analysis estimates dispersion using only the two selected groups; other studies and conditions are excluded from the model.
""", lang))
# ── 解析制御ロジック ──────────────────────────────────────────
            _meta_ctrl = st.session_state.get("metadata")
            _upload_mode_ctrl = st.session_state.get("upload_mode", "single")
            _min_samples_per_group = 0
            _group_sample_counts = {}
            if _meta_ctrl is not None:
                for _cond in st.session_state.get("conditions", []):
                    _n = (_meta_ctrl["condition"] == _cond).sum()
                    _group_sample_counts[_cond] = _n
                _min_samples_per_group = min(_group_sample_counts.values()) if _group_sample_counts else 0
            _deg_res_ctrl = st.session_state.get("deg_results")
            _n_sig_degs = 0
            if _deg_res_ctrl is not None:
                # BUG FIX: バグ① & バグ⑨ / A-1
                _n_sig_degs = (((_deg_res_ctrl["padj"].fillna(1.0) < padj_t) & (_deg_res_ctrl["log2FoldChange"].fillna(0.0).abs() > lfc_t))).sum()
            _n_studies = len(st.session_state.get("multi_study_names", []))

            conds = st.session_state["conditions"]
            _meta_deg = st.session_state.get("metadata")
            _upload_mode_deg = st.session_state.get("upload_mode", "single")

            if _upload_mode_deg == "multi" and _meta_deg is not None and "batch" in _meta_deg.columns:
                # Study別にグループ化して表示
                _study_cond_map = {}
                for _c in conds:
                    _batches = _meta_deg.loc[_meta_deg["condition"] == _c, "batch"].unique().tolist()
                    _b_label = _batches[0] if _batches else "Unknown"
                    if _b_label not in _study_cond_map:
                        _study_cond_map[_b_label] = []
                    _study_cond_map[_b_label].append(_c)

                # Study選択 → その中の群を選択
                _study_sel = st.selectbox(
                    ui('Select Study', lang, 'Study を選択 / Select Study'),
                    list(_study_cond_map.keys()),
                    key="deg_study_sel"
                )
                _conds_in_study = _study_cond_map[_study_sel]
                ref = st.selectbox(t("ref_group", lang), _conds_in_study, key="deg_ref_multi")
                test = st.selectbox(t("test_group", lang), [c for c in _conds_in_study if c != ref], key="deg_test_multi")

                if len(_conds_in_study) < 2:
                    st.warning("⚠️ " + (ui('This study has only one group. Please select another study.', lang, 'このStudyには群が1つしかありません。別のStudyを選んでください。')))
            else:
                ref = st.selectbox(t("ref_group", lang), conds)
                test = st.selectbox(t("test_group", lang), [c for c in conds if c != ref])
            
            max_cores = multiprocessing.cpu_count()
            if max_cores <= 1:
                n_cores = 1
                st.metric(ui("Number of CPU cores (Parallel Processing)", lang), 1)
            else:
                n_cores = st.slider(
                    ui("Number of CPU cores (Parallel Processing)", lang),
                    1, max_cores, min(4, max_cores),
                    help=ui("Multiple CPU cores can accelerate PyDESeq2 analysis.", lang),
                )
            st.session_state["n_cores"] = n_cores
            
            # サンプル数警告バッジ
            if _min_samples_per_group < 2:
                _err_msg = ""
                for _g, _n in _group_sample_counts.items():
                    if _n < 2: _err_msg += ui("Group '{value_0}' has only {value_1} samples. ", lang, '「{value_0}」群は{value_1}サンプルしかありません。', value_0=_g, value_1=_n)
                st.error("⛔ " + (ui('At least 2 samples per group are required. Currently, {value_0}Please check your group assignments in the Upload tab.', lang, '各群に最低2サンプル必要です。現在、{value_0}Uploadタブでサンプルの群割り当てを確認してください。', value_0=_err_msg)))
            elif _min_samples_per_group == 2:
                st.warning("⚠️ " + (ui("Only 2 samples per group. 3+ recommended (Cook's distance correction will be skipped).", lang, "各群2サンプルです。3サンプル以上を推奨します（Cook's distance補正がスキップされます）。")))
            # Multi Studyモードで異なるStudy間の比較警告
            if _upload_mode_ctrl == "multi" and _meta_ctrl is not None and "batch" in _meta_ctrl.columns:
                _ref_batch  = _meta_ctrl.loc[_meta_ctrl["condition"] == ref,  "batch"].unique().tolist() if ref  in _meta_ctrl["condition"].values else []
                _test_batch = _meta_ctrl.loc[_meta_ctrl["condition"] == test, "batch"].unique().tolist() if test in _meta_ctrl["condition"].values else []
                if _ref_batch and _test_batch and set(_ref_batch) != set(_test_batch):
                    st.error("⛔ " + (ui('Cross-study comparison is not recommended ({value_0} vs {value_1}). Please compare groups within the same study.', lang, '異なるStudy間の直接比較は推奨されません（{value_0} vs {value_1}）。同一Study内の群を選択してください。', value_0=_ref_batch[0], value_1=_test_batch[0])))
            _analyze_disabled = _min_samples_per_group < 2
            if st.button(ui("Analyze", lang), type="primary", disabled=_analyze_disabled):
                meta = st.session_state["metadata"]
                counts_per_group = meta[meta["condition"].isin([ref, test])]["condition"].value_counts()
                if counts_per_group.min() < 2:
                    _err_msg = ""
                    for _g, _n in counts_per_group.items():
                        if _n < 2: _err_msg += ui("Group '{value_0}' has only {value_1} samples. ", lang, '「{value_0}」群は{value_1}サンプルしかありません。', value_0=_g, value_1=_n)
                    st.error("⛔ " + (ui('At least 2 samples per group are required. Currently, {value_0}Please check your group assignments in the Upload tab.', lang, '各群に最低2サンプル必要です。現在、{value_0}Uploadタブでサンプルの群割り当てを確認してください。', value_0=_err_msg)))
                else:
                    try:
                        with st.status(ui("🎩 Running DEG analysis...", lang), expanded=True) as status:
                            data_to_analyze = active_counts_df()
                            res = run_deg(data_to_analyze, st.session_state["metadata"], ref, test, n_cpus=n_cores)
                            reset_contrast_results()
                            st.session_state["deg_results"] = res
                            st.session_state["last_contrast"] = f"{test} vs {ref}"
                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                        st.balloons()
                        st.success("✅ " + (ui("Analysis complete! Check results in the 'Visualization' or 'Network' tab.", lang, '解析完了！『Visualization』または『Network』タブで結果を確認してください。')))
                        st.rerun()
                    except Exception as e:
                        st.error("⛔ " + (ui('An error occurred during analysis. Common causes: (1) less than 2 samples per group, (2) negative counts, (3) special characters in sample names. Details: {value_0}', lang, '解析中にエラーが発生しました。よくある原因: (1)各群が2サンプル未満 (2)カウント値に負の数が含まれる (3)サンプル名に記号が含まれる。詳細: {value_0}', value_0=e)))
        with dr:
            if st.session_state["deg_results"] is not None:
                res = st.session_state["deg_results"]
                st.success(ui("Contrast: {contrast}", lang, contrast=st.session_state['last_contrast']))
                up = ((res["padj"] < padj_t) & (res["log2FoldChange"] > lfc_t)).sum()
                dn = ((res["padj"] < padj_t) & (res["log2FoldChange"] < -lfc_t)).sum()
                st.metric(ui("Up", lang), up); st.metric(ui("Down", lang), dn)
                st.dataframe(
                    res.head(100),
                    column_config={
                        "log2FoldChange": st.column_config.NumberColumn("log2FC", format="%.3f"),
                        "padj": st.column_config.NumberColumn("padj", format="%.2e"),
                        "pvalue": st.column_config.NumberColumn("p-value", format="%.2e"),
                        "stat": st.column_config.NumberColumn("stat", format="%.3f"),
                        "baseMean": st.column_config.NumberColumn("Base Mean", format="%.1f"),
                    },
                    width="stretch"
                )
                
                # Removed Volcano and MA plots from here (available in Visualization tab)

        # 複数コントラスト一括実行
        st.divider()
        _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
        if _is_jp:
            st.subheader("⚡ 複数コントラスト一括実行")
            st.markdown("実行するペアをチェックして「一括実行」ボタンを押してください。")
        else:
            st.subheader(ui("⚡ Batch DEG Analysis", lang))
            st.markdown(ui("Select the contrasts to run and click 'Run All'.", lang))

        _conds_batch = st.session_state.get("conditions", [])
        _all_pairs = [
            (r, t)
            for i, r in enumerate(_conds_batch)
            for t in _conds_batch[i+1:]
            if _group_sample_counts.get(r, 0) >= 2 and _group_sample_counts.get(t, 0) >= 2
        ]
        # Multi Studyモードの場合、同一Study内のペアのみに絞り込む
        if _upload_mode_ctrl == "multi" and _meta_ctrl is not None and "batch" in _meta_ctrl.columns:
            def _same_study(r, t):
                rb = _meta_ctrl.loc[_meta_ctrl["condition"] == r, "batch"].unique()
                tb = _meta_ctrl.loc[_meta_ctrl["condition"] == t, "batch"].unique()
                return set(rb) == set(tb)
            _all_pairs = [(_r, _t) for _r, _t in _all_pairs if _same_study(_r, _t)]
            if _is_jp:
                st.info("💡 Multi Studyモード：同一Study内のペアのみ表示しています。")
            else:
                st.info(ui("💡 Multi Study mode: showing only within-study pairs.", lang))
        if len(_all_pairs) >= 2:
            _selected_pairs = []
            _batch_cols = st.columns(min(3, len(_all_pairs)))
            for _pi, (_pr, _pt) in enumerate(_all_pairs):
                with _batch_cols[_pi % 3]:
                    if st.checkbox(f"{_pt} vs {_pr}", value=True, key=f"batch_chk_{_pr}_{_pt}"):
                        _selected_pairs.append((_pr, _pt))

            if st.button(ui('⚡ Run All Selected', lang, '⚡ 選択したペアを一括実行'), type="primary", key="batch_run_btn"):
                if not _selected_pairs:
                    st.warning(ui('Please select at least one pair.', lang, '少なくとも1つのペアを選択してください。/ Select at least one pair.'))
                else:
                    _batch_results = {}
                    _batch_provenance = {}
                    _data_batch = active_counts_df()
                    with st.status(ui("🎩 Running batch analysis for {count} pairs...", lang,
                                      count=len(_selected_pairs)), expanded=True) as _batch_status:
                        for _br, _bt in _selected_pairs:
                            st.write(f"⏳ {_bt} vs {_br}...")
                            try:
                                _res_batch = run_deg(_data_batch, st.session_state["metadata"], _br, _bt, n_cpus=st.session_state.get("n_cores", 1))
                                _contrast_key = f"{_bt}_vs_{_br}"
                                _batch_results[_contrast_key] = _res_batch
                                _contrast_studies = st.session_state["metadata"].loc[
                                    st.session_state["metadata"]["condition"].isin([_br, _bt]), "batch"
                                ].dropna().unique().tolist() if "batch" in st.session_state["metadata"].columns else []
                                _batch_provenance[_contrast_key] = _contrast_studies[0] if len(_contrast_studies) == 1 else None
                                _up = ((_res_batch["padj"] < padj_t) & (_res_batch["log2FoldChange"] > lfc_t)).sum()
                                _dn = ((_res_batch["padj"] < padj_t) & (_res_batch["log2FoldChange"] < -lfc_t)).sum()
                                st.write(f"✅ {_bt} vs {_br}: Up={_up}, Down={_dn}")
                                log_analysis("Batch DEG", f"Contrast: {_bt} vs {_br}, Up: {_up}, Down: {_dn}")
                            except Exception as _be:
                                st.write(f"❌ {_bt} vs {_br}: {_be}")
                        _batch_status.update(label="完了 / Done", state="complete", expanded=False)

                    st.session_state["batch_deg_results"] = _batch_results
                    st.session_state["batch_deg_provenance"] = _batch_provenance
                    st.session_state["lfc_meta_matrix"] = None
                    st.session_state["venn_deg_sets"] = None
                    st.session_state["venn_v_sel"] = None
                    st.session_state["venn_enr_kegg"] = None
                    st.session_state["venn_enr_go"] = None

            if st.session_state.get("batch_deg_results"):
                st.markdown("---")
                if _is_jp:
                    st.markdown("**一括実行結果のダウンロード**")
                else:
                    st.markdown(ui("**Download Batch Results**", lang))
                for _contrast_name, _res_dl in st.session_state["batch_deg_results"].items():
                    st.download_button(
                        f"📥 {_contrast_name}.csv",
                        _res_dl.to_csv(),
                        f"{_contrast_name}.csv",
                        key=f"dl_batch_{_contrast_name}"
                    )
                
                # UpSet Plot for batch DEG comparison
                if len(st.session_state["batch_deg_results"]) >= 2:
                    st.divider()
                    _up_title = ui('📊 UpSet Plot (DEG Overlap Across Contrasts)', lang, '📊 UpSet Plot（コントラスト間DEG比較）')
                    st.subheader(_up_title)
                    
                    _dir_label = ui('Direction filter', lang, '方向フィルター')
                    _dir_ids = ["up", "down", "both"]
                    _dir_labels = {
                        "up": ui("Up", lang),
                        "down": ui("Down", lang),
                        "both": ui("Both", lang),
                    }
                    _upset_dir = st.radio(
                        _dir_label, _dir_ids, format_func=_dir_labels.get,
                        horizontal=True, index=2, key="upset_dir"
                    )
                    
                    _upset_lbl = ui('Plot', lang, 'プロット')
                    if st.button(_upset_lbl, key="upset_run_btn"):
                        try:
                            from upsetplot import from_memberships, UpSet
                            
                            _upset_sets = {}
                            for _cn, _rd in st.session_state["batch_deg_results"].items():
                                _rd_clean = _rd.dropna(subset=["padj", "log2FoldChange"])
                                if _upset_dir == "up":
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"] > lfc_t)].index)
                                elif _upset_dir == "down":
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"] < -lfc_t)].index)
                                else:  # Both
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"].abs() > lfc_t)].index)
                                _upset_sets[_cn] = _genes
                            
                            # BUG FIX: バグ③
                            if not _upset_sets or all(len(v)==0 for v in _upset_sets.values()):
                                st.warning(ui('No DEGs found with current thresholds.', lang, '共通DEGが見つかりませんでした。'))
                                st.stop()

                            # Collect all genes with their membership
                            _all_genes = sorted(set.union(*_upset_sets.values()) if _upset_sets else set())
                            _memberships = [
                                [_cn for _cn, _gs in _upset_sets.items() if _gene in _gs]
                                for _gene in _all_genes
                            ]
                            _memberships = [m for m in _memberships if m]  # remove empty
                            
                            if _memberships:
                                _upset_data = from_memberships(_memberships)
                                _fig_upset, _ax_upset = plt.subplots(figsize=(12, 6))
                                UpSet(_upset_data, show_counts=True).plot(fig=_fig_upset)
                                st.pyplot(_fig_upset)
                                plt.close(_fig_upset)
                            else:
                                st.warning(ui('No DEGs found with current thresholds.', lang, '共通DEGが見つかりませんでした。'))
                        except ImportError:
                            st.error("pip install upsetplot が必要です / Please run: pip install upsetplot")
        else:
            if _is_jp:
                st.info("一括実行には3条件以上が必要です。")
            else:
                st.info(ui("Batch analysis requires 3 or more conditions.", lang))

        # Multi-group set comparison
        if st.session_state.get("conditions") and len(st.session_state["conditions"]) >= 3:
            st.divider()
            st.subheader(t("venn_title", lang))
            others = [c for c in st.session_state["conditions"] if c != ref]
            v_sel = st.multiselect(t("venn_groups_sel", lang), others, default=others[:2])

            # 方向フィルター（UP/DOWN/Both）
            _venn_dir_ids = ["up", "down", "both"]
            _venn_dir_labels = {
                "up": ui("Up only", lang),
                "down": ui("Down only", lang),
                "both": ui("Both (Up + Down)", lang),
            }
            _venn_dir = st.radio(
                ui('Direction filter', lang, '変動方向フィルター'),
                _venn_dir_ids,
                format_func=_venn_dir_labels.get,
                index=2,
                horizontal=True,
                key="venn_dir_filter",
                help=(
                    ui("Selecting 'Both' includes genes that are UP in one group and DOWN in another as shared DEGs. For KEGG/GO analysis, it is recommended to separate UP and DOWN genes.", lang, '「両方」を選ぶと、UPとDOWNが逆の遺伝子も共通DEGとして扱われます。KEGGやGO解析に使う場合はUP・DOWNを分けることを推奨します。')
                )
            )
            plot_type_venn = st.radio(ui("Plot type", lang), ["Venn", "UpSet"], horizontal=True)

            if st.session_state.get("upload_mode") == "multi":
                if st.session_state.get("batch_deg_results"):
                    if _is_jp:
                        st.info("💡 Multi Studyモード：一括実行済みのDEG結果を使用してVennを描画します。")
                    else:
                        st.info(ui("💡 Multi Study mode: Venn will use the already-computed batch DEG results.", lang))
                else:
                    if _is_jp:
                        st.warning("⚠️ 先に上の「Run All Selected」で各StudyのDEGを実行してください。")
                    else:
                        st.warning(ui("⚠️ Please run 'Run All Selected' above before drawing the Venn diagram.", lang))

            if st.button(t("venn_run_btn", lang)) and len(v_sel) >= 2:
                try:
                    _upload_mode_venn = st.session_state.get("upload_mode", "single")
                    _batch_res = st.session_state.get("batch_deg_results", {})

                    if _upload_mode_venn == "multi" and _batch_res:
                        # Multi Study mode: use already-computed batch_deg_results
                        # Match v_sel group names to contrast keys (e.g. "AD_vs_Control")
                        _deg_sets_new = {}
                        _missing = []
                        for _g in v_sel:
                            # Find a contrast key that starts with this group name
                            _matched_key = next(
                                (k for k in _batch_res if k.startswith(f"{_g}_vs_")), None
                            )
                            if _matched_key:
                                _res_vs = _batch_res[_matched_key]
                                if _venn_dir == "up":
                                    _venn_genes = set(_res_vs[
                                        (_res_vs["padj"] < padj_t) &
                                        (_res_vs["log2FoldChange"] > lfc_t)
                                    ].index)
                                elif _venn_dir == "down":
                                    _venn_genes = set(_res_vs[
                                        (_res_vs["padj"] < padj_t) &
                                        (_res_vs["log2FoldChange"] < -lfc_t)
                                    ].index)
                                else:
                                    _venn_genes = set(_res_vs[
                                        (_res_vs["padj"] < padj_t) &
                                        (_res_vs["log2FoldChange"].abs() > lfc_t)
                                    ].index)
                                _deg_sets_new[_g] = _venn_genes
                            else:
                                _missing.append(_g)

                        if _missing:
                            _warn_msg = (
                                ui("DEG results not found for: {value_0}. Please run 'Run All Selected' first.", lang, '以下のグループのDEG結果が見つかりません: {value_0}。先に「Run All Selected」で一括DEGを実行してください。', value_0=', '.join(_missing))
                            )
                            st.warning(_warn_msg)
                        if _deg_sets_new:
                            st.session_state["venn_deg_sets"] = _deg_sets_new
                            st.session_state["venn_v_sel"]    = list(_deg_sets_new.keys())
                    else:
                        # Single Study mode: run DEG on the fly as before
                        data_to_analyze = active_counts_df()
                        _deg_sets_new = {}
                        for g in v_sel:
                            _res_g = run_deg(data_to_analyze, st.session_state["metadata"], ref, g)
                            if _venn_dir == "up":
                                _deg_sets_new[g] = set(_res_g[
                                    (_res_g["padj"] < padj_t) &
                                    (_res_g["log2FoldChange"] > lfc_t)
                                ].index)
                            elif _venn_dir == "down":
                                _deg_sets_new[g] = set(_res_g[
                                    (_res_g["padj"] < padj_t) &
                                    (_res_g["log2FoldChange"] < -lfc_t)
                                ].index)
                            else:
                                _deg_sets_new[g] = set(_res_g[
                                    (_res_g["padj"] < padj_t) &
                                    (_res_g["log2FoldChange"].abs() > lfc_t)
                                ].index)
                        st.session_state["venn_deg_sets"] = _deg_sets_new
                        st.session_state["venn_v_sel"]    = v_sel

                except Exception as e:
                    st.error(ui("Venn/UpSet error: {error}", lang, error=e))

            # ── 描画・遺伝子テーブル・KEGG/GO ────────────────────────────
            if st.session_state.get("venn_deg_sets") and st.session_state.get("venn_v_sel"):
                from itertools import combinations as _combs
                _deg_sets = {k: v for k, v in st.session_state["venn_deg_sets"].items()
                             if k in st.session_state["venn_v_sel"]}
                _venn_groups = list(_deg_sets.keys())
                _all_union   = sorted(set.union(*_deg_sets.values())) if _deg_sets else []

                # ── Venn / UpSet 描画 ────────────────────────────────────
                if plot_type_venn == "Venn" and len(_deg_sets) <= 3:
                    import math as _math, random as _random
                    import plotly.graph_objects as _go_v
                    _random.seed(42)

                    _vc_pos = {
                        2: [(0.38, 0.5), (0.62, 0.5)],
                        3: [(0.38, 0.62), (0.62, 0.62), (0.5, 0.38)],
                    }
                    _vc_colors = [
                        ("rgba(255,120,120,0.35)", "rgba(255,80,80,0.8)"),
                        ("rgba(120,180,255,0.35)", "rgba(60,140,255,0.8)"),
                        ("rgba(120,220,120,0.35)", "rgba(60,180,60,0.8)"),
                    ]
                    _n_v = len(_deg_sets)
                    _positions = _vc_pos[_n_v]
                    _R = 0.21

                    _fig_vp = _go_v.Figure()

                    for _ci, (_gn, (_cx, _cy)) in enumerate(zip(_venn_groups, _positions)):
                        _th = [i * 2 * _math.pi / 100 for i in range(101)]
                        _fill, _line = _vc_colors[_ci]
                        _fig_vp.add_trace(_go_v.Scatter(
                            x=[_cx + _R * _math.cos(t) for t in _th],
                            y=[_cy + _R * _math.sin(t) for t in _th],
                            fill="toself", fillcolor=_fill,
                            line=dict(color=_line, width=2),
                            mode="lines", hoverinfo="skip",
                            showlegend=False,
                        ))
                        # BUG FIX: バグ② / A-2
                        _angle = [_math.pi*5/6, _math.pi/6, -_math.pi/2][_ci % 3]
                        _lx = _cx + (_R + 0.07) * _math.cos(_angle)
                        _ly = _cy + (_R + 0.07) * _math.sin(_angle)
                        _fig_vp.add_annotation(x=_lx, y=_ly, text=f"<b>{_gn}</b>",
                                               showarrow=False, font=dict(size=13))

                    _gx, _gy, _gnames, _gcols = [], [], [], []
                    _dot_colors = {
                        1: ["rgba(255,80,80,0.85)", "rgba(60,140,255,0.85)", "rgba(60,180,60,0.85)"],
                        2: "rgba(255,180,50,0.9)",
                        3: "rgba(180,80,220,0.9)",
                    }
                    for _gene in _all_union:
                        _in = [_g for _g in _venn_groups if _gene in _deg_sets[_g]]
                        _ni = len(_in)
                        if _ni == 1:
                            _idx = _venn_groups.index(_in[0])
                            _cx, _cy = _positions[_idx]
                            _angle = _random.uniform(0, 2 * _math.pi)
                            _rad   = _random.uniform(0.01, 0.10)
                            _gx.append(_cx + _rad * _math.cos(_angle))
                            _gy.append(_cy + _rad * _math.sin(_angle))
                            _gcols.append(_dot_colors[1][_idx])
                        elif _ni == _n_v:
                            _cx_c = sum(p[0] for p in _positions) / _n_v
                            _cy_c = sum(p[1] for p in _positions) / _n_v
                            _gx.append(_cx_c + _random.uniform(-0.03, 0.03))
                            _gy.append(_cy_c + _random.uniform(-0.03, 0.03))
                            _gcols.append(_dot_colors[3] if _n_v == 3 else _dot_colors[2])
                        else:
                            _idxs = [_venn_groups.index(_g) for _g in _in]
                            _cx_m = sum(_positions[_i][0] for _i in _idxs) / 2
                            _cy_m = sum(_positions[_i][1] for _i in _idxs) / 2
                            _gx.append(_cx_m + _random.uniform(-0.02, 0.02))
                            _gy.append(_cy_m + _random.uniform(-0.02, 0.02))
                            _gcols.append(_dot_colors[2])
                        _gnames.append(_gene)

                    _fig_vp.add_trace(_go_v.Scatter(
                        x=_gx, y=_gy, mode="markers",
                        marker=dict(size=9, color=_gcols, line=dict(width=1, color="white")),
                        text=_gnames,
                        hovertemplate="<b>%{text}</b><extra></extra>",
                        name="Genes",
                    ))

                    _fig_vp.update_layout(
                        xaxis=dict(visible=False, range=[0, 1]),
                        yaxis=dict(visible=False, range=[0, 1], scaleanchor="x"),
                        plot_bgcolor="white", height=500,
                        margin=dict(l=20, r=20, t=30, b=20),
                        template=plotly_template,
                    )
                    st.plotly_chart(_fig_vp, width="stretch")
                else:
                    try:
                        from upsetplot import from_memberships, UpSet
                        _memb = [[g for g in _deg_sets if gene in _deg_sets[g]] for gene in _all_union]
                        
                        if not _deg_sets or all(len(v) == 0 for v in _deg_sets.values()):
                            st.warning(ui('No DEGs found with current thresholds.', lang, '現在の閾値ではDEGが見つかりませんでした。')) # A-3
                        else:
                            _udata = from_memberships(_memb)
                            if len(_udata) > 0:
                                _fig_us, _ = plt.subplots(figsize=(12, 6))
                                UpSet(_udata, show_counts=True).plot(fig=_fig_us)
                                st.pyplot(_fig_us)
                                plt.close(_fig_us)
                    except ImportError:
                        st.error(ui("pip install upsetplot", lang))

                # ── 遺伝子所属マトリックス作成 ───────────────────────────
                _rows = []
                for _gene in _all_union:
                    _row = {"Gene": _gene}
                    for _g in _venn_groups:
                        _row[_g] = "✅" if _gene in _deg_sets[_g] else ""
                    _in_groups = [_g for _g in _venn_groups if _gene in _deg_sets[_g]]
                    _row["Groups"] = " & ".join(_in_groups)
                    _rows.append(_row)
                _member_df = pd.DataFrame(_rows)

                # ── 領域フィルタ UI ──────────────────────────────────────
                st.divider()
                st.markdown("#### 🧬 " + (ui('Gene List — select a region to run KEGG / GO', lang, '遺伝子一覧 / 領域を選んでKEGG・GO解析が可能')))

                _filter_opts = (
                    ["All"] +
                    _venn_groups +
                    [" & ".join(sorted([_a, _b])) for _a, _b in _combs(_venn_groups, 2)] +
                    ([" & ".join(sorted(_venn_groups))] if len(_venn_groups) == 3 else [])
                )
                # ラベルを日本語/英語で
                _filter_labels = {
                    "All": ui('All genes', lang, 'All（全遺伝子）'),
                    " & ".join(sorted(_venn_groups)): "🔴 " + (ui('Common to all 3 groups', lang, '3群共通')) if len(_venn_groups) == 3 else "",
                }
                _filter_sel = st.selectbox(
                    ui('Select region', lang, '領域を選択 / Select region'),
                    _filter_opts,
                    format_func=lambda x: _filter_labels.get(x, x),
                    key="venn_gene_filter"
                )

                # フィルタ適用
                if _filter_sel == "All":
                    _filtered_df = _member_df
                elif " & " in _filter_sel:
                    _sel_groups = _filter_sel.split(" & ")
                    # 選択した群全員に属する遺伝子（かつ他群には属さない場合はExclusive、Allは共通）
                    _filtered_df = _member_df[
                        _member_df["Groups"] == " & ".join(sorted(_sel_groups))
                    ]
                else:
                    _filtered_df = _member_df[_member_df[_filter_sel] == "✅"]

                st.caption(f"{ui('Selected', lang, '選択中')}: **{_filter_sel}** — {len(_filtered_df)} genes")
                st.dataframe(_filtered_df, width="stretch", height=250)

                # CSVダウンロード
                st.download_button(
                    "📥 " + (ui('Download gene list (CSV)', lang, '遺伝子一覧をCSVでダウンロード')),
                    _filtered_df.to_csv(index=False),
                    "venn_gene_list.csv", "text/csv",
                    key="dl_venn_genes"
                )

                # ── 選択領域でKEGG / GO ─────────────────────────────────
                st.divider()
                _sel_genes = _filtered_df["Gene"].tolist()
                _n_sel = len(_sel_genes)

                if _n_sel == 0:
                    st.warning("⚠️ " + (ui('No genes in this region.', lang, 'この領域に遺伝子がありません。')))
                else:
                    st.markdown(f"#### 🔬 " + (ui('Pathway Analysis for: {value_0} — {value_1} genes', lang, '選択領域（{value_0}）の経路解析 — {value_1} genes', value_0=_filter_sel, value_1=_n_sel)))
                    _venn_enr_col1, _venn_enr_col2 = st.columns(2)

                    # KEGG
                    with _venn_enr_col1:
                        if st.button(ui("🧬 KEGG", lang), type="primary", key="venn_kegg_btn", width="stretch"):
                            try:
                                with st.status(ui("🎩 Running KEGG analysis...", lang)):
                                    st.session_state["venn_enr_kegg"] = run_overrepresentation(
                                        _sel_genes,
                                        st.session_state["sp"]["gene_sets_kegg"],
                                        active_counts_df().index.tolist(),
                                    )
                            except Exception as _e:
                                st.error(ui("KEGG error: {error}", lang, error=_e))

                        if st.session_state.get("venn_enr_kegg") is not None:
                            _vk_df = st.session_state["venn_enr_kegg"].head(10)
                            _fig_vk = px.bar(
                                _vk_df, x="Combined Score", y="Term", orientation="h",
                                color="Adjusted P-value",
                                color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"),
                                title=f"Top 10 KEGG ({_filter_sel})",
                                template=plotly_template
                            )
                            _fig_vk.update_layout(yaxis={"categoryorder": "total ascending", "title": ""}, font=dict(size=11))
                            st.plotly_chart(_fig_vk, width="stretch")
                            st.download_button(
                                ui("📥 KEGG CSV", lang), _vk_df.to_csv(index=False),
                                "venn_kegg.csv", "text/csv", key="dl_venn_kegg"
                            )

                    # GO
                    with _venn_enr_col2:
                        if st.button(ui("🌿 GO", lang), type="primary", key="venn_go_btn", width="stretch"):
                            try:
                                with st.status(ui("🎩 Running GO analysis...", lang)):
                                    st.session_state["venn_enr_go"] = run_overrepresentation(
                                        _sel_genes,
                                        st.session_state["sp"]["gene_sets_go"],
                                        active_counts_df().index.tolist(),
                                    )
                            except Exception as _e:
                                st.error(ui("GO error: {error}", lang, error=_e))

                        if st.session_state.get("venn_enr_go") is not None:
                            _vg_df = st.session_state["venn_enr_go"].head(10)
                            _fig_vg = px.bar(
                                _vg_df, x="Combined Score", y="Term", orientation="h",
                                color="Adjusted P-value",
                                color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"),
                                title=f"Top 10 GO ({_filter_sel})",
                                template=plotly_template
                            )
                            _fig_vg.update_layout(yaxis={"categoryorder": "total ascending", "title": ""}, font=dict(size=11))
                            st.plotly_chart(_fig_vg, width="stretch")
                            st.download_button(
                                ui("📥 GO CSV", lang), _vg_df.to_csv(index=False),
                                "venn_go.csv", "text/csv", key="dl_venn_go"
                            )

    # ── Interaction Analysis (Advanced) ─────────────────────────
    st.divider()
    with st.expander(
        ui('🔬 Interaction Analysis (Advanced)', lang, '🔬 交互作用解析（Advanced）'),
        expanded=False
    ):
        if _is_jp:
            st.markdown("""
ある変数（例: age, batch）によって、別の変数（treatment）の効果がどのように変調されるかを解析します。

**デザイン式:** `~ var1 + condition + var1:condition`

**必要条件:**
- メタデータに交互作用変数が必要（連続数値またはカテゴリ変数に対応）
- 各群に4サンプル以上を推奨
- 出力は交互作用項（`var1:condition`）の結果のみ表示
""")
        else:
            st.markdown(ui("""
Identifies genes whose response to treatment is modulated by another variable (e.g., age, batch).

**Design formula:** `~ var1 + condition + var1:condition`

**Requirements:**
- Metadata must contain an interaction variable (continuous numeric or categorical)
- At least 4 samples per group recommended
- Output shows only the interaction term results
""", lang))

        _meta_ia = st.session_state.get("metadata")
        _counts_ia = active_counts_df()

        if _meta_ia is None or _counts_ia is None:
            st.info("💡 " + (ui('Please upload data in the Upload tab first.', lang, 'まずUploadタブでデータをアップロードしてください。')))
        else:
            _interaction_reserved_columns = {"condition", "batch", "sample_label"}
            _extra_cols = [c for c in _meta_ia.columns if c not in _interaction_reserved_columns]
            if not _extra_cols:
                st.warning("⚠️ " + (
                    ui("Metadata must have columns other than 'condition'. Please add an interaction variable (e.g., age).", lang, "メタデータに'condition'以外の列が必要です。交互作用変数（age等）を追加してください。")
                ))
            else:
                _ia_col1, _ia_col2 = st.columns(2)
                with _ia_col1:
                    _ia_var = st.selectbox(
                        ui('Interaction variable', lang, '交互作用変数'),
                        _extra_cols,
                        key="ia_var_sel"
                    )
                    # 変数タイプの自動判定（数値かどうか）
                    _ia_var_vals = _meta_ia[_ia_var].dropna()
                    try:
                        _ia_var_vals.astype(float)
                        _ia_var_is_numeric_default = True
                    except (ValueError, TypeError):
                        _ia_var_is_numeric_default = False

                    _ia_var_type_opts = (
                        ui(['Continuous (numeric)', 'Categorical (factor)'], lang, ['連続変数（numeric）', 'カテゴリ変数（factor）'])
                    )
                    _ia_var_type = st.radio(
                        ui('Variable type', lang, '変数タイプ'),
                        _ia_var_type_opts,
                        index=0 if _ia_var_is_numeric_default else 1,
                        horizontal=True,
                        key="ia_var_type"
                    )
                    _ia_use_numeric = _ia_var_type == _ia_var_type_opts[0]

                    # カテゴリ変数の場合のみ参照水準選択UIを表示
                    if not _ia_use_numeric:
                        _ia_var_levels = sorted(_meta_ia[_ia_var].dropna().astype(str).unique().tolist())
                        _ia_ref_level = st.selectbox(
                            ui('Reference level of interaction variable', lang, '参照水準（Reference level）'),
                            _ia_var_levels,
                            index=0,
                            key="ia_ref_level_sel",
                            help=(
                                ui("This level is used as the baseline. Other levels are compared against it. Choose the biologically meaningful baseline (e.g., 'young').", lang, 'この水準を基準として他の水準との交互作用効果を計算します。生物学的に「対照」となる群（例: young）を選んでください。')
                            )
                        )
                        # 参照水準以外の水準を表示（これらが係数として生成される）
                        _ia_other_levels = [l for l in _ia_var_levels if l != _ia_ref_level]
                        _ia_coefficients = ", ".join(
                            [f'`{_ia_var}[T.{level}]:condition[T.{{test}}]`' for level in _ia_other_levels]
                        )
                        st.caption(ui("📌 Coefficients to be generated: {coefficients}", lang,
                                      coefficients=_ia_coefficients))
                    else:
                        _ia_ref_level = None

                with _ia_col2:
                    _ia_conditions = st.session_state.get("conditions", [])
                    if (
                        st.session_state.get("upload_mode") == "multi"
                        and "batch" in _meta_ia.columns
                    ):
                        _ia_studies = _meta_ia["batch"].dropna().unique().tolist()
                        _ia_selected_study = st.selectbox(
                            ui("Select Study", lang), _ia_studies, key="ia_study_sel"
                        )
                        _ia_conditions = _meta_ia.loc[
                            _meta_ia["batch"] == _ia_selected_study, "condition"
                        ].dropna().unique().tolist()
                    _ia_ref = st.selectbox(
                        ui("Reference condition", lang),
                        _ia_conditions,
                        key="ia_ref_sel"
                    )
                    _ia_test = st.selectbox(
                        ui("Test condition", lang),
                        [c for c in _ia_conditions if c != _ia_ref],
                        key="ia_test_sel"
                    )

                _design_preview = f"~ {_ia_var} + condition + {_ia_var}:condition"
                st.code(_design_preview, language="r")

                # 変数タイプに応じた注意書き
                if _ia_use_numeric:
                    st.info(
                        ui('📐 **Continuous mode**: The variable is treated as numeric. Tests whether the treatment effect changes linearly with the variable (e.g., age). Suitable for detecting monotonic trends.', lang, '📐 **連続変数モード**: 年齢などを数値のまま扱い、「処置効果が変数の増加とともに線形に変化するか」を検定します。単調増加・減少トレンドの検出に適しています。')
                    )
                else:
                    st.info(
                        ui('🏷️ **Categorical mode**: The variable is treated as discrete groups. Tests whether the treatment effect differs between groups. Suitable when group boundaries are biologically meaningful.', lang, '🏷️ **カテゴリ変数モード**: 変数を独立した群として扱い、「各群で処置効果が異なるか」を検定します。生物学的に意味のある区切りがある場合に適しています。')
                    )

                _ia_meta_selected = _meta_ia.loc[
                    _meta_ia["condition"].isin([_ia_ref, _ia_test]), ["condition", _ia_var]
                ].dropna()
                _ia_can_run = True
                if _ia_use_numeric:
                    _ia_group_counts = _ia_meta_selected["condition"].value_counts()
                    _ia_group_variation = _ia_meta_selected.groupby("condition")[_ia_var].nunique()
                    _ia_can_run = (
                        len(_ia_group_counts) == 2
                        and int(_ia_group_counts.min()) >= 4
                        and len(_ia_group_variation) == 2
                        and int(_ia_group_variation.min()) >= 2
                    )
                    if not _ia_can_run:
                        st.error(ui("Continuous interaction analysis requires at least 4 samples and at least 2 distinct numeric values in each condition.", lang))
                    elif int(_ia_group_counts.min()) < 6:
                        st.warning(ui("At least 6 samples per condition are recommended for continuous interaction analysis.", lang))
                else:
                    _ia_cell_counts = pd.crosstab(
                        _ia_meta_selected["condition"], _ia_meta_selected[_ia_var].astype(str)
                    ).reindex(index=[_ia_ref, _ia_test], fill_value=0)
                    _ia_can_run = (
                        _ia_cell_counts.shape[1] >= 2
                        and not _ia_cell_counts.empty
                        and int(_ia_cell_counts.min().min()) >= 2
                    )
                    if not _ia_can_run:
                        st.error(ui("Categorical interaction analysis requires at least 2 samples in every condition × level cell and at least 2 levels.", lang))
                    elif int(_ia_cell_counts.min().min()) < 3:
                        st.warning(ui("At least 3 samples in every condition × level cell are recommended.", lang))

                _ia_padj = st.number_input(
                    ui('padj threshold', lang, 'padj閾値'),
                    0.0, 1.0, 0.05, 0.005, key="ia_padj_t"
                )
                _ia_lfc = st.number_input(
                    ui('LFC threshold', lang, 'LFC閾値'),
                    0.0, 5.0, 0.5, 0.1, key="ia_lfc_t"
                )

                if st.button(
                    ui('▶ Run Interaction Analysis', lang, '▶ 交互作用解析を実行'),
                    type="primary", key="ia_run_btn", disabled=not _ia_can_run
                ):
                    try:
                        from pydeseq2.dds import DeseqDataSet
                        from pydeseq2.ds import DeseqStats

                        _meta_sub = _meta_ia[_meta_ia["condition"].isin([_ia_ref, _ia_test])].copy()
                        _counts_sub = _counts_ia.T.loc[_meta_sub.index]
                        _meta_sub["condition"] = pd.Categorical(
                            _meta_sub["condition"].astype(str),
                            categories=[str(_ia_ref), str(_ia_test)],
                        )

                        # 変数タイプに応じて変換を切り替える
                        if _ia_use_numeric:
                            # 連続変数：数値型のまま渡す（傾きの線形変化を検定）
                            try:
                                _meta_sub[_ia_var] = _meta_sub[_ia_var].astype(float)
                            except (ValueError, TypeError):
                                st.error(
                                    ui("⛔ Could not convert '{value_0}' to numeric. Switch to Categorical mode or ensure all values are numbers.", lang, '⛔ 「{value_0}」を数値に変換できませんでした。カテゴリ変数モードに切り替えるか、値を数値にしてください。', value_0=_ia_var)
                                )
                                st.stop()
                        else:
                            # カテゴリ変数：参照水準を明示的に指定してCategorical型に変換
                            # pandas Categorical の categories の最初の要素が参照水準になる
                            _ia_var_str = _meta_sub[_ia_var].astype(str)
                            _all_levels = sorted(_ia_var_str.unique().tolist())
                            if _ia_ref_level and str(_ia_ref_level) in _all_levels:
                                # 指定した参照水準を先頭に、残りをアルファベット順で並べる
                                _ordered_levels = [str(_ia_ref_level)] + [
                                    l for l in _all_levels if l != str(_ia_ref_level)
                                ]
                            else:
                                # フォールバック：アルファベット順
                                _ordered_levels = _all_levels
                            _meta_sub[_ia_var] = pd.Categorical(
                                _ia_var_str,
                                categories=_ordered_levels
                            )

                        with st.status(
                            ui('🎩 Running interaction analysis...', lang, '🎩 交互作用解析中...'),
                            expanded=True
                        ):
                            _dds_ia = DeseqDataSet(
                                counts=_counts_sub,
                                metadata=_meta_sub,
                                design=f"~ {_ia_var} + condition + {_ia_var}:condition",
                                refit_cooks=True,
                                n_cpus=st.session_state.get("n_cores", 1)
                            )
                            _dds_ia.deseq2()

                            # デザイン行列の列名から全係数名を取得
                            # dds.obsm["design_matrix"].columns が公式の取得方法
                            try:
                                _all_coef_names = list(_dds_ia.obsm["design_matrix"].columns)
                            except Exception:
                                # フォールバック: result_names() を試みる
                                try:
                                    _all_coef_names = list(_dds_ia.result_names())
                                except Exception:
                                    _all_coef_names = []

                            if not _all_coef_names:
                                st.error(
                                    ui('⛔ Could not retrieve design matrix coefficient names.', lang, '⛔ デザイン行列の係数名を取得できませんでした。')
                                )
                                st.stop()

                            st.write("🔍 利用可能な係数名 / Available coefficients:")
                            st.code("\n".join(_all_coef_names))

                            # 交互作用項の係数名を抽出
                            # PyDESeq2の命名規則例:
                            #   カテゴリ: "age[T.old]:condition[T.Treatment]"
                            #   連続変数: "age:condition[T.Treatment]"
                            # 変数名とconditionを両方含む係数を交互作用項として抽出
                            # Interceptと純粋な主効果は除外する
                            _ia_coef_names = [
                                c for c in _all_coef_names
                                if (
                                    _ia_var.lower() in c.lower()
                                    and "condition" in c.lower()
                                    and "intercept" not in c.lower()
                                )
                            ]

                            # 連続変数モードでも交互作用項が見つからない場合はエラーで止める
                            # （主効果へのフォールバックは「交互作用解析」の名称と矛盾するため）

                            if not _ia_coef_names:
                                st.error(
                                    ui('⛔ No interaction term coefficients found. Available coefficients: {value_0}', lang, '⛔ 交互作用項の係数が見つかりませんでした。利用可能な係数: {value_0}', value_0=_all_coef_names)
                                )
                                st.stop()

                            # 各交互作用項を個別に検定
                            _ia_results_dict = {}
                            for _coef in _ia_coef_names:
                                try:
                                    # PyDESeq2の name引数で交互作用係数を直接指定
                                    _stat_ia = DeseqStats(
                                        _dds_ia,
                                        name=_coef,
                                        n_cpus=st.session_state.get("n_cores", 1)
                                    )
                                    _stat_ia.summary()
                                    _res_coef = _stat_ia.results_df.copy()
                                    _res_coef["padj"] = _res_coef["padj"].fillna(1.0)
                                    _res_coef["log2FoldChange"] = _res_coef["log2FoldChange"].fillna(0.0)
                                    _ia_results_dict[_coef] = _res_coef
                                    st.write(ui("✅ Coefficient: {coefficient}", lang, coefficient=_coef))
                                except Exception as _coef_e:
                                    st.warning(f"⚠️ {_coef}: {_coef_e}")

                            if not _ia_results_dict:
                                st.error(
                                    ui('⛔ All interaction term tests failed.', lang, '⛔ 全ての交互作用項の検定に失敗しました。')
                                )
                                st.stop()

                            # 全項の結果を保存
                            # 表示用の代表結果：項が1つならそのまま、複数なら最初の項を代表とする
                            # （各項は後のUI選択で切り替え可能）
                            _first_coef = list(_ia_results_dict.keys())[0]
                            st.session_state["ia_results"] = _ia_results_dict[_first_coef]
                            st.session_state["ia_all_term_results"] = _ia_results_dict
                            st.session_state["ia_coef_names"] = list(_ia_results_dict.keys())
                            log_analysis(
                                "Interaction Analysis",
                                f"Var: {_ia_var}, Terms: {list(_ia_results_dict.keys())}, Contrast: {_ia_test} vs {_ia_ref}"
                            )

                        st.rerun()

                    except Exception as _ia_e:
                        st.error(ui("Interaction analysis error: {error}", lang, error=_ia_e))

                if st.session_state.get("ia_results") is not None:
                    # 複数交互作用項がある場合は選択UIを表示
                    _ia_coef_list = st.session_state.get("ia_coef_names", [])
                    _ia_all_terms = st.session_state.get("ia_all_term_results", {})
                    if len(_ia_coef_list) > 1:
                        _selected_term = st.selectbox(
                            ui('Select interaction term to display', lang, '表示する交互作用項を選択'),
                            _ia_coef_list,
                            key="ia_term_sel"
                        )
                        st.caption(
                            ui('📌 Showing: `{value_0}`  \nEach term represents the interaction effect at a different level of the variable.', lang, '📌 現在表示中: `{value_0}`  \n各項は異なる年齢水準での交互作用効果を表します。', value_0=_selected_term)
                        )
                        _res_show = _ia_all_terms.get(_selected_term, st.session_state["ia_results"])
                    else:
                        _res_show = st.session_state["ia_results"]
                    _sig_ia = _res_show[
                        (_res_show["padj"] < _ia_padj) &
                        (_res_show["log2FoldChange"].abs() > _ia_lfc)
                    ]
                    _ia_up = (_sig_ia["log2FoldChange"] > 0).sum()
                    _ia_dn = (_sig_ia["log2FoldChange"] < 0).sum()

                    _r1, _r2, _r3 = st.columns(3)
                    _r1.metric(ui("Significant genes", lang), len(_sig_ia))
                    _r2.metric(ui("Up", lang), _ia_up)
                    _r3.metric(ui("Down", lang), _ia_dn)

                    st.dataframe(
                        _sig_ia.sort_values("padj"),
                        column_config={
                            "padj": st.column_config.NumberColumn("padj", format="%.2e"),
                            "pvalue": st.column_config.NumberColumn("p-value", format="%.2e"),
                            "log2FoldChange": st.column_config.NumberColumn("log2FC", format="%.3f"),
                        },
                        width="stretch"
                    )
                    st.download_button(
                        "📥 " + (ui('Download interaction results CSV', lang, '交互作用解析結果 CSV')),
                        _sig_ia.to_csv(),
                        "interaction_results.csv",
                        key="dl_ia_btn"
                    )

                    # ── 可視化セクション ─────────────────────────────────
                    st.divider()
                    st.markdown("### 📊 " + (ui('Visualization', lang, '可視化')))

                    _viz_tabs = st.tabs([
                        "🔥 " + (ui('Heatmap', lang, 'ヒートマップ')),
                        "📈 " + (ui('Interaction Line Plot', lang, 'Interaction Line Plot')),
                        "🎯 " + (ui('LFC Scatter', lang, 'LFC Scatter')),
                    ])

                    _norm_ia = normalize_counts(
                        st.session_state["counts_df"],
                        st.session_state.get("norm_method", "log1p"),
                        st.session_state.get("gene_lengths")
                    )
                    _meta_viz = st.session_state["metadata"]
                    _ia_var_viz = st.session_state.get("ia_var_sel", "")

                    # ── Tab 1: Heatmap ────────────────────────────────────
                    with _viz_tabs[0]:
                        if len(_sig_ia) == 0:
                            st.info(ui('No significant genes found.', lang, '有意な遺伝子がありません。'))
                        else:
                            if len(_sig_ia) <= 5:
                                _top_n_ia = len(_sig_ia)
                                st.metric(ui('Top N genes', lang, '表示遺伝子数'), _top_n_ia)
                            else:
                                _top_n_ia = st.slider(
                                    ui('Top N genes', lang, '表示遺伝子数'),
                                    5, min(50, len(_sig_ia)), min(20, len(_sig_ia)),
                                    key="ia_hm_n"
                                )
                            _top_genes_ia = _sig_ia.sort_values("padj").head(_top_n_ia).index.tolist()
                            _top_genes_ia = [g for g in _top_genes_ia if g in _norm_ia.index]

                            # サンプルをグループ順に並べる
                            _sample_order = []
                            if _ia_var_viz and _ia_var_viz in _meta_viz.columns:
                                for _v in sorted(_meta_viz[_ia_var_viz].unique()):
                                    for _c in [_ia_ref, _ia_test]:
                                        _mask = (
                                            (_meta_viz["condition"] == _c) &
                                            (_meta_viz[_ia_var_viz] == _v)
                                        )
                                        _sample_order += _meta_viz[_mask].index.tolist()
                            else:
                                _sample_order = _meta_viz.index.tolist()

                            _hm_data = _norm_ia.loc[_top_genes_ia, _sample_order]
                            _hm_z = _hm_data.subtract(_hm_data.mean(axis=1), axis=0).divide(
                                _hm_data.std(axis=1).replace(0, 1), axis=0
                            )
                            _fig_ia_hm = px.imshow(
                                _hm_z,
                                aspect="auto",
                                color_continuous_scale="RdBu_r",
                                color_continuous_midpoint=0,
                                template=plotly_template,
                                title="Interaction Heatmap (Z-score)",
                                labels={"x": "Sample", "y": "Gene", "color": "Z-score"}
                            )
                            _fig_ia_hm.update_layout(
                                font=dict(family=st.session_state.get("selected_font", "sans-serif")),
                                height=max(400, len(_top_genes_ia) * 18)
                            )
                            st.plotly_chart(_fig_ia_hm, width="stretch")

                    # ── Tab 2: Interaction Line Plot ──────────────────────
                    with _viz_tabs[1]:
                        if len(_sig_ia) == 0:
                            st.info(ui('No significant genes found.', lang, '有意な遺伝子がありません。'))
                        else:
                            # なげなわ選択遺伝子があればそれをデフォルトに使う
                            _lasso_default = st.session_state.get("ia_lasso_genes", [])
                            _valid_lasso = [g for g in _lasso_default if g in _sig_ia.index]
                            _fallback_default = _sig_ia.sort_values("padj").head(4).index.tolist()
                            _line_default = _valid_lasso[:8] if _valid_lasso else _fallback_default
                            _line_genes_key = f"ia_line_genes_{st.session_state.get('ia_line_genes_key', 0)}"
                            _line_genes = st.multiselect(
                                ui('Select genes (max 8)', lang, '遺伝子を選択（最大8個）'),
                                _sig_ia.sort_values("padj").index.tolist(),
                                default=_line_default,
                                max_selections=8,
                                key=_line_genes_key
                            )
                            if _line_genes and _ia_var_viz and _ia_var_viz in _meta_viz.columns:
                                _line_rows = []
                                for _g in _line_genes:
                                    if _g not in _norm_ia.index:
                                        continue
                                    for _s in _meta_viz.index:
                                        _line_rows.append({
                                            "Gene": _g,
                                            "Sample": _s,
                                            "Expression": _norm_ia.loc[_g, _s],
                                            "condition": _meta_viz.loc[_s, "condition"],
                                            _ia_var_viz: str(_meta_viz.loc[_s, _ia_var_viz])
                                        })
                                _line_df = pd.DataFrame(_line_rows)
                                _line_mean = _line_df.groupby(
                                    ["Gene", "condition", _ia_var_viz]
                                )["Expression"].mean().reset_index()

                                _fig_ia_line = px.line(
                                    _line_mean,
                                    x="condition",
                                    y="Expression",
                                    color=_ia_var_viz,
                                    facet_col="Gene",
                                    facet_col_wrap=min(4, len(_line_genes)),
                                    markers=True,
                                    template=plotly_template,
                                    title="Interaction Line Plot (mean expression)",
                                    category_orders={"condition": [_ia_ref, _ia_test]}
                                )
                                _fig_ia_line.update_layout(
                                    font=dict(family=st.session_state.get("selected_font", "sans-serif")),
                                    height=400 if len(_line_genes) <= 4 else 700
                                )
                                st.plotly_chart(_fig_ia_line, width="stretch")
                            else:
                                st.info(
                                    ui('Select genes and ensure interaction variable is set.', lang, '遺伝子と交互作用変数を選択してください。')
                                )

                    # ── Tab 3: LFC Scatter ────────────────────────────────
                    with _viz_tabs[2]:
                        if _ia_var_viz and _ia_var_viz in _meta_viz.columns:
                            _var_vals_unique = sorted(_meta_viz[_ia_var_viz].unique())
                            if len(_var_vals_unique) >= 2:
                                from itertools import combinations as _lfc_combs
                                _lfc_pairs = list(_lfc_combs(_var_vals_unique, 2))

                                # 全遺伝子・全カテゴリのLFCを事前計算
                                _lfc_all = {}
                                for _vval in _var_vals_unique:
                                    _vval_str = str(_vval)
                                    _lfc_all[_vval_str] = {}
                                    for _g in _res_show.index:
                                        if _g not in _norm_ia.index:
                                            continue
                                        _ctrl_s = _meta_viz[
                                            (_meta_viz["condition"] == _ia_ref) &
                                            (_meta_viz[_ia_var_viz] == _vval)
                                        ].index
                                        _trt_s = _meta_viz[
                                            (_meta_viz["condition"] == _ia_test) &
                                            (_meta_viz[_ia_var_viz] == _vval)
                                        ].index
                                        if len(_ctrl_s) > 0 and len(_trt_s) > 0:
                                            _lfc_all[_vval_str][_g] = (
                                                _norm_ia.loc[_g, _trt_s].mean() -
                                                _norm_ia.loc[_g, _ctrl_s].mean()
                                            )

                                st.caption(
                                    "💡 " + (
                                        ui('Lasso or box-select dots to send selected genes to the Line Plot. Choose lasso or box tool from the top-right toolbar.', lang, 'なげなわ/ボックスでドットを囲むと選択遺伝子がLine Plotに反映されます。右上のツールバーでlassoまたはboxを選択してください。')
                                    )
                                )

                                # セッションステートの初期化
                                if "ia_lasso_genes" not in st.session_state:
                                    st.session_state["ia_lasso_genes"] = []

                                for (_va, _vb) in _lfc_pairs:
                                    _va_str = str(_va)
                                    _vb_str = str(_vb)
                                    _col_va = f"LFC_{_va_str}"
                                    _col_vb = f"LFC_{_vb_str}"
                                    _genes_both = [
                                        _g for _g in _res_show.index
                                        if _g in _lfc_all.get(_va_str, {}) and _g in _lfc_all.get(_vb_str, {})
                                    ]
                                    if not _genes_both:
                                        continue
                                    _lfc_df = pd.DataFrame({
                                        "Gene": _genes_both,
                                        _col_va: [_lfc_all[_va_str][_g] for _g in _genes_both],
                                        _col_vb: [_lfc_all[_vb_str][_g] for _g in _genes_both],
                                    })
                                    _lfc_df["significant"] = _lfc_df["Gene"].isin(_sig_ia.index)
                                    _fig_lfc = px.scatter(
                                        _lfc_df,
                                        x=_col_va,
                                        y=_col_vb,
                                        color="significant",
                                        hover_name="Gene",
                                        color_discrete_map={True: "#E64B35", False: "#7f8c8d"},
                                        template=plotly_template,
                                        title=f"LFC Scatter: {_ia_var_viz} {_va_str} vs {_vb_str}",
                                        labels={
                                            _col_va: f"LFC ({_ia_var_viz}={_va_str})",
                                            _col_vb: f"LFC ({_ia_var_viz}={_vb_str})"
                                        }
                                    )
                                    _lim = max(
                                        _lfc_df[_col_va].abs().max(),
                                        _lfc_df[_col_vb].abs().max()
                                    ) * 1.1
                                    _fig_lfc.add_shape(
                                        type="line",
                                        x0=-_lim, y0=-_lim,
                                        x1=_lim, y1=_lim,
                                        line=dict(dash="dash", color="gray", width=1)
                                    )
                                    _fig_lfc.update_layout(
                                        font=dict(family=st.session_state.get("selected_font", "sans-serif")),
                                        height=500,
                                        dragmode="lasso"
                                    )

                                    # streamlit-plotly-eventsが使える場合はなげなわ選択を有効化
                                    if _HAS_PLOTLY_EVENTS:
                                        _selected_points = _plotly_events(
                                            _fig_lfc,
                                            select_event=True,
                                            override_height=520,
                                            key=f"lasso_{_va_str}_{_vb_str}"
                                        )
                                        if _selected_points:
                                            _selected_genes = []
                                            for _pt in _selected_points:
                                                _pt_idx = _pt.get("pointIndex")
                                                if _pt_idx is not None and _pt_idx < len(_lfc_df):
                                                    _selected_genes.append(_lfc_df.iloc[_pt_idx]["Gene"])
                                            if _selected_genes:
                                                st.session_state["ia_lasso_genes"] = _selected_genes
                                                st.session_state["ia_line_genes_key"] = st.session_state.get("ia_line_genes_key", 0) + 1
                                                st.rerun()
                                    else:
                                        st.plotly_chart(_fig_lfc, width="stretch")
                                        if st.session_state.get("ia_lasso_genes"):
                                            st.info(
                                                ui('💡 To enable lasso selection, run `pip install streamlit-plotly-events`.', lang, '💡 なげなわ選択を有効にするには `pip install streamlit-plotly-events` を実行してください。')
                                            )

                                    st.divider()

                                # なげなわ選択遺伝子のクリア
                                if st.session_state.get("ia_lasso_genes"):
                                    if st.button(
                                        ui('🗑️ Clear selection', lang, '🗑️ 選択をクリア'),
                                        key="ia_lasso_clear"
                                    ):
                                        st.session_state["ia_lasso_genes"] = []
                                        st.rerun()

                        else:
                            st.info(
                                ui('No interaction variable set. Please configure additional variables in the Upload tab.', lang, '交互作用変数が設定されていません。Uploadタブで追加変数を設定してください。')
                            )


# TAB 3: VISUALIZATION
with tab_viz:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["deg_results"] is None:
        if _is_jp:
            st.info("💡 **解析結果がありません**\n\nまずは **DEG** タブで差次発現解析（Analyze）を実行してください。")
        else:
            st.info(ui("💡 **Empty State**\n\nPlease run differential expression analysis in the **DEG** tab first.", lang))
    else:

        def render_plot_settings(key_suffix):
            with st.expander(ui("⚙️ Plot Settings", lang), expanded=False):
                _dynamic_fonts = list(dict.fromkeys([app_font_name] + FONTS))
                f_c1, f_c2 = st.columns(2)
                with f_c1:
                    sel_font = st.selectbox(ui("Font", lang), _dynamic_fonts, index=_dynamic_fonts.index(st.session_state.get("selected_font", "sans-serif")), key=f"viz_font_{key_suffix}")
                    img_format = st.selectbox(ui("Export format", lang), ["png", "pdf", "svg"], index=["png", "pdf", "svg"].index(st.session_state.get("img_format", "png")), key=f"viz_fmt_{key_suffix}")
                    img_dpi = st.selectbox(ui("DPI", lang), [300, 600], index=[300, 600].index(st.session_state.get("img_dpi", 300)), key=f"viz_dpi_{key_suffix}")
                with f_c2:
                    up_color = st.color_picker("Up color", st.session_state.get("up_color", "#E64B35"), key=f"viz_upc_{key_suffix}")
                    down_color = st.color_picker("Down color", st.session_state.get("down_color", "#4DBBD5"), key=f"viz_dnc_{key_suffix}")
                fig_width = st.slider(ui("Width (px)", lang), 400, 1600, st.session_state.get("fig_width", 800), 50, key=f"viz_w_{key_suffix}")
                fig_height = st.slider(ui("Height (px)", lang), 300, 1200, st.session_state.get("fig_height", 500), 50, key=f"viz_h_{key_suffix}")
                fig_font_sz = st.slider(ui("Font size (pt)", lang), 8, 28, st.session_state.get("fig_font_sz", 12), 1, key=f"viz_sz_{key_suffix}")
                
                st.session_state["selected_font"] = sel_font
                st.session_state["img_format"] = img_format
                st.session_state["img_dpi"] = img_dpi
                st.session_state["up_color"] = up_color
                st.session_state["down_color"] = down_color
                st.session_state["fig_width"] = fig_width
                st.session_state["fig_height"] = fig_height
                st.session_state["fig_font_sz"] = fig_font_sz

                if st.session_state.get("conditions"):
                    with st.expander(ui("Group Colors", lang), expanded=False):
                        _conds = st.session_state["conditions"]
                        _palette = sns.color_palette("husl", len(_conds)).as_hex()
                        _custom_colors = st.session_state.get("custom_cond_colors", {})
                        for i, cond in enumerate(_conds):
                            _custom_colors[cond] = st.color_picker(f"Color for {cond}", _custom_colors.get(cond, _palette[i]), key=f"viz_c_{cond}_{key_suffix}")
                        st.session_state["custom_cond_colors"] = _custom_colors
                
                with st.expander(ui("Analysis Plot Colors", lang), expanded=False):
                    _ecmap_opts = ["Viridis_r", "Plasma_r", "Magma_r", "Inferno_r", "Cividis_r"]
                    _hcmap_opts = ["RdBu_r", "Spectral_r", "Coolwarm", "RdYlBu_r"]
                    enr_cmap = st.selectbox(ui("Enrichment Scale", lang), _ecmap_opts, index=_ecmap_opts.index(st.session_state.get("enr_cmap", "Viridis_r")), key=f"viz_ecmap_{key_suffix}")
                    hm_cmap = st.selectbox(ui("Heatmap Scale", lang), _hcmap_opts, index=_hcmap_opts.index(st.session_state.get("hm_cmap", "RdBu_r")), key=f"viz_hcmap_{key_suffix}")
                    st.session_state["enr_cmap"] = enr_cmap
                    st.session_state["hm_cmap"] = hm_cmap
            return sel_font, img_format, img_dpi, up_color, down_color, fig_width, fig_height, fig_font_sz

        lfc_t, padj_t = st.session_state["deg_t"]
        v_tab, m_tab, h_tab, gp_tab = st.tabs(
            [ui("Volcano", lang), ui("MA Plot", lang), ui("Heatmap", lang), ui("Gene Plot", lang)]
        )
        def get_img_bytes(fig, fmt, dpi):
            """Plotly図を画像バイト列に変換。"""
            try:
                return fig.to_image(format=fmt, scale=dpi/72)
            except Exception as e:
                st.error(ui("Export error: {error}", lang, error=e))
                return None


        plotly_config = {
            'toImageButtonOptions': {
                'format': img_format,
                'filename': 'bulk_rnaseq_plot',
                'height': None,
                'width': None,
                'scale': img_dpi / 72
            },
            'displaylogo': False
        }

        with v_tab:
            sel_font, img_format, img_dpi, up_color, down_color, fig_width, fig_height, fig_font_sz = render_plot_settings("v")
            # 遺伝子検索（Volcanoタブ内に移動）
            with st.expander("🔍 " + (ui('Highlight a Gene', lang, '遺伝子をハイライト')), expanded=False):
                sc1, sc2 = st.columns([3, 1])
                all_genes = sorted(st.session_state["counts_df"].index.tolist())
                q_gene = sc1.selectbox(t("gene_search_placeholder", lang), [""] + all_genes, key="search_q")
                if sc2.button(t("search_btn", lang), type="primary") and q_gene:
                    st.session_state["viz_highlight"] = q_gene
            with st.expander(ui("About Volcano Plot", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
Volcano Plot は log2FoldChange（x軸）と統計的有意性 −log10(padj)（y軸）を同時に示します。右上が有意な上昇発現、左上が有意な下降発現です。

- 縦・横の破線がそれぞれ padj 閾値・LFC 閾値を示します
- 遺伝子検索で特定の遺伝子を星印でハイライトできます
- 点をホバーすると遺伝子名・統計値を確認できます
""")
                else:
                    st.markdown(ui("""
The Volcano Plot displays log2FoldChange (x-axis) and statistical significance −log10(padj) (y-axis) simultaneously. Genes in the upper-right are significantly up-regulated; upper-left are significantly down-regulated.

- Dashed lines indicate the padj threshold (horizontal) and LFC threshold (vertical)
- Use the gene search box to highlight a specific gene with a star marker
- Hover over any point to view gene name and statistics
""", lang))
            hl = st.session_state.get("search_q", "")
            fig_v = plot_volcano_plotly(st.session_state["deg_results"], padj_t, lfc_t, up_color, down_color, template=plotly_template, font=sel_font, highlight_gene=hl, font_size=fig_font_sz)
            fig_v.update_layout(width=fig_width, height=fig_height, font=dict(family=sel_font, size=fig_font_sz))
            try:
                event = st.plotly_chart(fig_v, width="stretch", config=plotly_config, on_select="rerun")
                sel_pts = []
                if isinstance(event, dict):
                    sel_pts = event.get("selection", {}).get("points", [])
                elif hasattr(event, "selection"):
                    sel_pts = getattr(event.selection, "points", [])
                
                sel_genes = []
                for pt in sel_pts:
                    if isinstance(pt, dict) and "hovertext" in pt: sel_genes.append(pt["hovertext"])
                    elif hasattr(pt, "hovertext"): sel_genes.append(pt.hovertext)
                
                if sel_genes:
                    st.success(ui("🎯 Lasso selection ({count} genes): {genes}...", lang,
                                  count=len(sel_genes), genes=", ".join(sel_genes[:10])))
                    st.session_state["custom_gene_list"] = sel_genes
            except TypeError:
                st.plotly_chart(fig_v, width="stretch", config=plotly_config)
            img_v = get_img_bytes(fig_v, img_format, img_dpi)
            if img_v: st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img_v, f"volcano.{img_format}", key="dl_v_btn")
        
        with m_tab:
            sel_font, img_format, img_dpi, up_color, down_color, fig_width, fig_height, fig_font_sz = render_plot_settings("m")
            with st.expander(ui("About MA Plot", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
MA Plot は平均発現量（baseMean、x軸・log スケール）と log2FoldChange（y軸）の関係を示します。

- 低発現遺伝子ほどLFCがばらつく傾向があり、フィルタリングの参考になります
- y=0 の破線からの乖離が大きいほど発現変動が大きい遺伝子です
- Volcano Plot と併用することで解析の信頼性を確認できます
""")
                else:
                    st.markdown(ui("""
The MA Plot shows mean expression (baseMean, x-axis, log scale) vs. log2FoldChange (y-axis).

- Low-expression genes tend to show higher LFC variance — useful for filtering decisions
- Greater deviation from the y=0 dashed line indicates larger expression change
- Use alongside the Volcano Plot to verify analysis reliability
""", lang))
            hl = st.session_state.get("search_q", "")
            fig_m = plot_ma_plotly(st.session_state["deg_results"], padj_t, lfc_t, up_color, down_color, template=plotly_template, font=sel_font, highlight_gene=hl, font_size=fig_font_sz)
            fig_m.update_layout(width=fig_width, height=fig_height, font=dict(family=sel_font, size=fig_font_sz))
            st.plotly_chart(fig_m, width="stretch", config=plotly_config)
            img_m = get_img_bytes(fig_m, img_format, img_dpi)
            if img_m: st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img_m, f"ma_plot.{img_format}", key="dl_m_btn")



        # — NEW: Heatmap tab
        with h_tab:
            sel_font, img_format, img_dpi, up_color, down_color, fig_width, fig_height, fig_font_sz = render_plot_settings("h")
            with st.expander(ui("About DEG Heatmap", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
有意DEGの上位N遺伝子について、全サンプルの正規化発現量（現在の設定に連動）を表示します。推奨: **log1p** または **VST**。

- 表示遺伝子数はスライダーで調整できます（デフォルト: 上位50遺伝子）
- padj でソートされた上位遺伝子が選択されます
- 高解像度なFigureとして出力可能です
""")
                else:
                    st.markdown(ui("""
Displays normalized expression of the top N significant DEGs across all samples (linked to current normalization setting). Recommended: **log1p** or **VST**.

- Number of genes shown is adjustable via slider (default: top 50 genes)
- Genes are selected by sorting on padj (most significant first)
- Output can be saved as high-resolution figures
""", lang))
            res_h = st.session_state["deg_results"]
            norm_cnt = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"), st.session_state.get("gene_lengths"))
            top_n_h = st.slider(ui("Top N genes", lang), 10, 200, 50, key="heatmap_n")
            
            # --- 追加: 表示モードの選択 ---
            _hm_mode_ids = ["sample", "group"]
            _hm_mode_labels = {
                "sample": ui("Per Sample", lang),
                "group": ui("Group Average", lang),
            }
            hm_mode = st.radio(
                ui("Display Mode", lang), _hm_mode_ids, format_func=_hm_mode_labels.get,
                horizontal=True, key="hm_display_mode"
            )
            
            # Select top N by padj
            top_genes = res_h.dropna(subset=["padj"]).sort_values("padj").head(top_n_h).index
            top_genes = [g for g in top_genes if g in norm_cnt.index]
            if top_genes:
                hm_df = norm_cnt.loc[top_genes]
                
                # --- 追加: 群平均計算ロジック ---
                if hm_mode == "group":
                    meta_h = st.session_state["metadata"]
                    hm_df = hm_df.T.groupby(meta_h["condition"]).mean().T

                # Z-score normalise per gene
                hm_z = hm_df.subtract(hm_df.mean(axis=1), axis=0).divide(hm_df.std(axis=1).replace(0, 1), axis=0)
                _hm_norm_label_map = {
                    "log1p": "log1p(CPM)-normalised",
                    "CPM":   "CPM-normalised",
                    "TPM":   "TPM-normalised",
                    "VST":   "VST-normalised",
                }
                _hm_norm_label = _hm_norm_label_map.get(
                    st.session_state.get("norm_method", "log1p"), "normalised"
                )
                fig_h = px.imshow(
                    hm_z,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    template=plotly_template,
                    title=f"Top {top_n_h} DEGs — Z-scored {_hm_norm_label} counts",
                    labels={"x": "Sample", "y": "Gene", "color": "Z-score"},
                )
                fig_h.update_layout(
                    font=dict(family=sel_font, size=fig_font_sz),
                    width=fig_width, height=max(fig_height, top_n_h * 14)
                )
                fig_h.update_yaxes(tickfont_size=max(6, 130 // top_n_h))
                st.plotly_chart(fig_h, width="stretch", config=plotly_config)
                img_h = get_img_bytes(fig_h, img_format, img_dpi)
                if img_h:
                    st.download_button(ui("📥 Download heatmap ({format})", lang, format=img_format.upper()),
                                       img_h, f"heatmap.{img_format}", key="dl_h_btn")
            else:
                st.info("ℹ️ " + (ui("No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun.", lang, '有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。')))

        # — NEW: Gene Plot tab
        with gp_tab:
            sel_font, img_format, img_dpi, up_color, down_color, fig_width, fig_height, fig_font_sz = render_plot_settings("gp")
            with st.expander(ui("About Gene Plot", lang), expanded=False):
                if _is_jp:
                    st.markdown("""複数の遺伝子を選択してパネル表示（Facet）が可能です。最大12遺伝子まで推奨。

**推奨正規化方法:** log1p または CPM
- **log1p**: 外れ値の影響を抑えた汎用的な可視化に適しています
- **CPM**: ライブラリサイズの違いが大きいサンプル間比較に適しています
""")
                else:
                    st.markdown(ui("""Select multiple genes for panel (facet) display. Up to 12 genes recommended.

**Recommended normalization:** log1p or CPM
- **log1p**: suitable for general visualization, reduces the effect of outliers
- **CPM**: suitable for comparing samples with large differences in library size
""", lang))

            norm_gp = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"), st.session_state.get("gene_lengths"))
            meta_gp = st.session_state["metadata"]
            all_g_list = sorted(norm_gp.index.tolist())
            
            gp_genes = st.multiselect(ui("Select genes", lang), all_g_list,
                                      default=[st.session_state.get("search_q")] if st.session_state.get("search_q") in all_g_list else [all_g_list[0]] if all_g_list else [],
                                      max_selections=12, key="gp_genes_multisel")
            
            _gp_type_ids = ["box", "violin"]
            _gp_type_labels = {
                "box": ui("Boxplot", lang),
                "violin": ui("Violin", lang),
            }
            gp_type = st.radio(
                ui("Plot type", lang), _gp_type_ids, format_func=_gp_type_labels.get,
                horizontal=True, key="gp_type_radio"
            )
            
            if gp_genes:
                # NCBI Links (Chip-style for all selected genes)
                _sp_name = st.session_state.get("sp", {}).get("org", "mouse")
                links_html = "".join([
                    f'<a href="https://www.ncbi.nlm.nih.gov/gene/?term={g}+[{_sp_name}]" target="_blank" style="text-decoration: none;">'
                    f'<span style="background-color: rgba(79, 110, 247, 0.1); color: #4F6EF7; padding: 2px 10px; border-radius: 15px; margin-right: 6px; font-size: 12px; border: 1px solid rgba(79, 110, 247, 0.2); display: inline-block; margin-bottom: 8px; font-weight: 500;">'
                    f'{g} ↗️</span></a>'
                    for g in gp_genes
                ])
                st.markdown(f'<div style="margin-top: 10px; margin-bottom: 5px;">{links_html}</div>', unsafe_allow_html=True)

                # Data preparation (Melt)
                expr_data = norm_gp.loc[gp_genes].T.reset_index()
                expr_data.columns = ["Sample"] + gp_genes
                expr_data["condition"] = expr_data["Sample"].map(meta_gp["condition"])

                _norm_label_map = {
                    "log1p": "log1p(CPM)",
                    "CPM":   "CPM",
                    "TPM":   "TPM",
                    "VST":   "VST-normalized counts",
                }
                _y_label = _norm_label_map.get(
                    st.session_state.get("norm_method", "log1p"), "Normalized counts"
                )

                melted = expr_data.melt(id_vars=["Sample", "condition"], var_name="Gene", value_name=_y_label)
                
                cond_palette = st.session_state.get("custom_cond_colors", {})
                
                # Dynamic height based on number of genes (rows)
                n_rows = (len(gp_genes) - 1) // 3 + 1
                dynamic_height = max(fig_height, 300 * n_rows)

                if gp_type == "box":
                    fig_gp = px.box(melted, x="condition", y=_y_label, color="condition",
                                    facet_col="Gene", facet_col_wrap=3, facet_row_spacing=0.1,
                                    points="all", hover_data=["Sample"],
                                    color_discrete_map=cond_palette, template=plotly_template,
                                    title="Multi-Gene Panel Plot")
                else:
                    fig_gp = px.violin(melted, x="condition", y=_y_label, color="condition",
                                       facet_col="Gene", facet_col_wrap=3,
                                       box=True, points="all", hover_data=["Sample"],
                                       color_discrete_map=cond_palette, template=plotly_template,
                                       title="Multi-Gene Panel Plot")

                fig_gp.update_layout(font=dict(family=sel_font, size=fig_font_sz), 
                                     width=fig_width, height=dynamic_height, showlegend=True)
                fig_gp.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>")) # 遺伝子名を太字に
                
                st.plotly_chart(fig_gp, width="stretch", config=plotly_config)
                img_gp = get_img_bytes(fig_gp, img_format, img_dpi)
                if img_gp:
                    file_name_gp = f"gene_plot_panel.{img_format}"
                    st.download_button(ui("📥 Download ({format})", lang, format=img_format.upper()),
                                       img_gp, file_name_gp, key="dl_gp_btn")

# TAB 4: NETWORK
with tab_network:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["deg_results"] is None:
        if _is_jp:
            st.info("💡 **解析結果がありません**\n\nまずは **DEG** タブで差次発現解析（Analyze）を実行してください。")
        else:
            st.info(ui("💡 **Empty State**\n\nPlease run differential expression analysis in the **DEG** tab first.", lang))
    else:
        # --- 🎨 Figure Settings (Tab-local) ---
        with st.expander(ui("🎨 Figure Settings", lang), expanded=False):
            _dynamic_fonts = list(dict.fromkeys([app_font_name] + FONTS))
            f_c1, f_c2 = st.columns(2)
            with f_c1:
                sel_font = st.selectbox(ui("Font", lang), _dynamic_fonts, index=_dynamic_fonts.index(st.session_state.get("selected_font", "sans-serif")), key="net_font")
                img_format = st.selectbox(ui("Export format", lang), ["png", "pdf", "svg"], index=["png", "pdf", "svg"].index(st.session_state.get("img_format", "png")), key="net_fmt")
                img_dpi = st.selectbox(ui("DPI", lang), [300, 600], index=[300, 600].index(st.session_state.get("img_dpi", 300)), key="net_dpi")
            with f_c2:
                up_color = st.color_picker("Up color", st.session_state.get("up_color", "#E64B35"), key="net_upc")
                down_color = st.color_picker("Down color", st.session_state.get("down_color", "#4DBBD5"), key="net_dnc")
            fig_width = st.slider(ui("Width (px)", lang), 400, 1600, st.session_state.get("fig_width", 800), 50, key="net_w")
            fig_height = st.slider(ui("Height (px)", lang), 300, 1200, st.session_state.get("fig_height", 500), 50, key="net_h")
            fig_font_sz = st.slider(ui("Font size (pt)", lang), 8, 28, st.session_state.get("fig_font_sz", 12), 1, key="net_sz")
            
            st.session_state["selected_font"] = sel_font
            st.session_state["img_format"] = img_format
            st.session_state["img_dpi"] = img_dpi
            st.session_state["up_color"] = up_color
            st.session_state["down_color"] = down_color
            st.session_state["fig_width"] = fig_width
            st.session_state["fig_height"] = fig_height
            st.session_state["fig_font_sz"] = fig_font_sz

            if st.session_state.get("conditions"):
                with st.expander(ui("Group Colors", lang), expanded=False):
                    _conds = st.session_state["conditions"]
                    _palette = sns.color_palette("husl", len(_conds)).as_hex()
                    _custom_colors = st.session_state.get("custom_cond_colors", {})
                    for i, cond in enumerate(_conds):
                        _custom_colors[cond] = st.color_picker(f"Color for {cond}", _custom_colors.get(cond, _palette[i]), key=f"net_c_{cond}")
                    st.session_state["custom_cond_colors"] = _custom_colors
            
            with st.expander(ui("Analysis Plot Colors", lang), expanded=False):
                _ecmap_opts = ["Viridis_r", "Plasma_r", "Magma_r", "Inferno_r", "Cividis_r"]
                _hcmap_opts = ["RdBu_r", "Spectral_r", "Coolwarm", "RdYlBu_r"]
                enr_cmap = st.selectbox(ui("Enrichment Scale", lang), _ecmap_opts, index=_ecmap_opts.index(st.session_state.get("enr_cmap", "Viridis_r")), key="net_ecmap")
                hm_cmap = st.selectbox(ui("Heatmap Scale", lang), _hcmap_opts, index=_hcmap_opts.index(st.session_state.get("hm_cmap", "RdBu_r")), key="net_hcmap")
                st.session_state["enr_cmap"] = enr_cmap
                st.session_state["hm_cmap"] = hm_cmap

        # Use variables from session state
        sel_font = st.session_state["selected_font"]
        img_format = st.session_state["img_format"]
        img_dpi = st.session_state["img_dpi"]
        up_color = st.session_state["up_color"]
        down_color = st.session_state["down_color"]
        fig_width = st.session_state["fig_width"]
        fig_height = st.session_state["fig_height"]
        fig_font_sz = st.session_state["fig_font_sz"]

        st.header(ui("Network & Functional Analysis", lang))
        # Request 2: Reorganize into Category Radios
        nt1_k, nt1_g, nt1_gsea, nt2, nt_corr, nt3, nt4 = st.tabs([
            ui("🔬 KEGG", lang), ui("🔬 GO", lang), ui("🔬 GSEA", lang),
            ui("🔗 STRING", lang), ui("🔗 Correlation", lang),
            ui("🧫 TF Activity", lang), ui("🧫 Deconvolution", lang)
        ])

        _gene_set_filter_ids = ["all", "up", "down", "custom"]
        _gene_set_filter_labels = {
            "all": ui("All", lang),
            "up": ui("UP Only", lang),
            "down": ui("DOWN Only", lang),
            "custom": ui("Lasso Selected (Custom)", lang),
        }

        with nt1_k:
            st.subheader(ui("KEGG Pathway Enrichment", lang))
            with st.expander(ui("About KEGG enrichment", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
**KEGG (Kyoto Encyclopedia of Genes and Genomes)** は代謝経路・シグナル伝達経路を収録した代表的なデータベースです。  
選択した有意DEGリストをKEGG経路と照合し、**Enrichr (GSEApy)** を用いて過剰表現解析を行います。

- **Combined Score** = ln(*p*-value) × z-score（大きいほど有意）
- **Adjusted P-value** で補正済み
- Up/Down/All の方向で遺伝子セットを絞り込めます
""")
                else:
                    st.markdown(ui("""
**KEGG (Kyoto Encyclopedia of Genes and Genomes)** is a major database of metabolic and signaling pathways. Significant DEGs are tested for over-representation against KEGG pathway gene sets using **Enrichr (GSEApy)**.

- **Combined Score** = ln(p-value) × z-score (higher = more significant)
- **Adjusted P-value** is multiple-testing corrected
- Gene sets can be filtered by direction: Up / Down / All
""", lang))
            direction_k = st.selectbox(
                ui("Gene set", lang), _gene_set_filter_ids,
                format_func=_gene_set_filter_labels.get, key="k_dir"
            )
            if _n_sig_degs == 0:
                st.info("ℹ️ " + (ui("No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun.", lang, '有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。')))
            _kegg_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if st.button(ui("Fetch Pathways", lang), type="primary", disabled=_kegg_disabled):
                try:
                    res_deg = st.session_state["deg_results"]
                    if direction_k == "up":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange > {lfc_t}").index.tolist()
                    elif direction_k == "down":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange < {-lfc_t}").index.tolist()
                    elif direction_k == "custom":
                        sig_genes = st.session_state.get("custom_gene_list", [])
                    else:
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()

                    if not sig_genes:
                        st.warning(ui("No significant genes found for the selected direction.", lang))
                    else:
                        with st.status(ui("🎩 Running KEGG analysis...", lang)):
                            st.session_state["enr_kegg"] = run_overrepresentation(
                                sig_genes,
                                st.session_state["sp"]["gene_sets_kegg"],
                                active_counts_df().index.tolist(),
                            )
                        log_analysis("KEGG Run", f"Direction: {direction_k}, Genes: {len(sig_genes)}")
                except Exception as e: st.error(ui("KEGG error: {error}", lang, error=e))
            
            if st.session_state["enr_kegg"] is not None:
                st.divider()
                k_c1, k_c2 = st.columns(2)
                with k_c1:
                    top_n_k = st.slider(t("top_n_paths", lang), 5, 50, 10, key="k_top_n")
                with k_c2:
                    pt_k = st.selectbox(t("plot_type", lang), [t("bar_plot", lang), t("dot_plot", lang)], key="k_pt")
                
                df = st.session_state["enr_kegg"].head(top_n_k)
                _direction_k_label = _gene_set_filter_labels[direction_k]
                if pt_k == t("bar_plot", lang):
                    fig = px.bar(df, x='Combined Score', y='Term', orientation='h', title=f"Top {top_n_k} KEGG ({_direction_k_label})",
                                 color='Adjusted P-value', color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"), template=plotly_template)
                    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=sel_font, size=fig_font_sz))
                else:
                    fig = plot_enrich_dot_plotly(df, f"Top {top_n_k} KEGG ({_direction_k_label})", plotly_template, sel_font, fig_font_sz)
                
                st.plotly_chart(fig, width="stretch", config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"kegg_plot.{img_format}", key="dl_kegg_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(ui("### KEGG Data Table (Top {count})", lang, count=top_n_k))
                st.dataframe(
                    df,
                    column_config={
                        "Adjusted P-value": st.column_config.NumberColumn("padj", format="%.2e"),
                        "P-value": st.column_config.NumberColumn("p-value", format="%.2e"),
                        "Odds Ratio": st.column_config.NumberColumn(format="%.2f"),
                        "Combined Score": st.column_config.NumberColumn(format="%.2f"),
                    },
                    width="stretch"
                )
        with nt1_g:
            st.subheader(ui("GO Biological Process", lang))
            with st.expander(ui("About GO enrichment", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
**GO (Gene Ontology) Biological Process** は遺伝子が関与する生物学的プロセスを階層的に分類したオントロジーです。  
**Enrichr (GSEApy)** を用いて過剰表現解析（ORA）を実行します。

- KEGGより粒度が細かく、細胞内シグナルや分子機能レベルの解釈に適しています
- 同様にCombined Score / Adjusted P-valueで評価
- Up/Down/All の方向でフィルタ可能
""")
                else:
                    st.markdown(ui("""
**GO (Gene Ontology) Biological Process** is a hierarchical ontology classifying the biological processes genes are involved in. Over-representation analysis (ORA) is performed using **Enrichr (GSEApy)**.

- Provides finer resolution than KEGG — well-suited for interpreting intracellular signaling and molecular functions
- Results are evaluated using Combined Score and Adjusted P-value
- Gene sets can be filtered by direction: Up / Down / All
""", lang))
            direction_g = st.selectbox(
                ui("Gene set", lang), _gene_set_filter_ids,
                format_func=_gene_set_filter_labels.get, key="g_dir"
            )
            if _n_sig_degs == 0:
                st.info("ℹ️ " + (ui("No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun.", lang, '有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。')))
            _go_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if st.button(ui("Fetch GO Terms", lang), type="primary", disabled=_go_disabled):
                try:
                    res_deg = st.session_state["deg_results"]
                    if direction_g == "up":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange > {lfc_t}").index.tolist()
                    elif direction_g == "down":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange < {-lfc_t}").index.tolist()
                    elif direction_g == "custom":
                        sig_genes = st.session_state.get("custom_gene_list", [])
                    else:
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()

                    if not sig_genes:
                        st.warning(ui("No significant genes found for the selected direction.", lang))
                    else:
                        with st.status(ui("🎩 Running GO analysis...", lang)):
                            st.session_state["enr_go"] = run_overrepresentation(
                                sig_genes,
                                st.session_state["sp"]["gene_sets_go"],
                                active_counts_df().index.tolist(),
                            )
                        log_analysis("GO Run", f"Direction: {direction_g}, Genes: {len(sig_genes)}")
                except Exception as e: st.error(ui("GO error: {error}", lang, error=e))
            
            if st.session_state["enr_go"] is not None:
                st.divider()
                g_c1, g_c2 = st.columns(2)
                with g_c1:
                    top_n_g = st.slider(t("top_n_terms", lang), 5, 50, 10, key="g_top_n")
                with g_c2:
                    pt_g = st.selectbox(t("plot_type", lang), [t("bar_plot", lang), t("dot_plot", lang)], key="g_pt")
                
                df = st.session_state["enr_go"].head(top_n_g)
                _direction_g_label = _gene_set_filter_labels[direction_g]
                if pt_g == t("bar_plot", lang):
                    fig = px.bar(df, x='Combined Score', y='Term', orientation='h', title=f"Top {top_n_g} GO ({_direction_g_label})",
                                 color='Adjusted P-value', color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"), template=plotly_template)
                    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=sel_font, size=fig_font_sz))
                else:
                    fig = plot_enrich_dot_plotly(df, f"Top {top_n_g} GO ({_direction_g_label})", plotly_template, sel_font, fig_font_sz)
                
                st.plotly_chart(fig, width="stretch", config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"go_plot.{img_format}", key="dl_go_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(ui("### GO Data Table (Top {count})", lang, count=top_n_g))
                st.dataframe(df, width="stretch")
        with nt1_gsea:
            st.subheader(ui("Preranked GSEA", lang))
            with st.expander(ui("About GSEA", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
**Gene Set Enrichment Analysis (GSEA)** は、全遺伝子を −log10(p値) × sign(log2FC) でランク付けし、遺伝子セット内の遺伝子が上位または下位に集中しているかを評価します。  
ORA（過剰表現解析）とは異なり、閾値で切り捨てることなく全遺伝子を使用するため、微弱だが一貫したシグナルを検出できます。

- **NES (Normalized Enrichment Score)**: 正値＝上位ランク（up-regulated）、負値＝下位ランク（down-regulated）
- **FDR q-value**: ≤ 0.25 が一般的な有意閾値
- 本実装は **GSEApy prerank** を使用し、ランク指標には −log10(p値) × sign(log2FC) を採用（統計的有意性と効果量を両方反映）
""")
                else:
                    st.markdown(ui("""
**Gene Set Enrichment Analysis (GSEA)** ranks all genes by −log10(p-value) × sign(log2FC) and tests whether genes in a gene set are concentrated at the top or bottom of the ranking. Unlike ORA, no threshold cutoff is applied — all genes are used, enabling detection of weak but consistent signals.

- **NES (Normalized Enrichment Score)**: positive = enriched at top (up-regulated); negative = enriched at bottom (down-regulated)
- **FDR q-value**: ≤ 0.25 is the conventional significance threshold
- This implementation uses **GSEApy prerank** with KEGG gene sets, ranked by −log10(p-value) × sign(log2FC), reflecting both statistical significance and effect direction
""", lang))
            _gsea_disabled = _deg_res_ctrl is None
            if _gsea_disabled:
                st.warning("⚠️ " + (ui('Please run DEG analysis first.', lang, '先にDEG解析を実行してください。')))
            if st.button(ui("Calculate GSEA", lang), type="primary", disabled=_gsea_disabled):
                try:
                    import gseapy as gp
                    res_deg = st.session_state["deg_results"]
                    # ランクスコア: -log10(pvalue) × sign(log2FC)
                    # Wald statのみより統計的有意性と効果量を両方反映できる
                    res_deg_rank = res_deg.copy()
                    res_deg_rank['pvalue'] = res_deg_rank['pvalue'].replace(0, 1e-300).fillna(1.0)
                    res_deg_rank['log2FoldChange'] = res_deg_rank['log2FoldChange'].fillna(0.0)
                    res_deg_rank['score'] = (
                        -np.log10(res_deg_rank['pvalue']) * 
                        np.sign(res_deg_rank['log2FoldChange'])
                    )
                    rank = res_deg_rank[['score']].sort_values('score', ascending=False).reset_index()
                    rank.columns = ['gene_name', 'score']
                    
                    with st.status(ui("🎩 Running GSEA...", lang), expanded=True):
                        # prerank実行（詳細プロットのために結果オブジェクトを丸ごと保存）
                        pre_res = gp.prerank(
                            rnk=rank,
                            gene_sets=resolve_gene_set(st.session_state["sp"]["gene_sets_kegg"]),
                            outdir=None,
                        )
                        st.session_state["gsea_results"] = pre_res.res2d
                        st.session_state["gsea_object"] = pre_res
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(ui("GSEA error: {error}", lang, error=e))
            
            if st.session_state["gsea_results"] is not None:
                st.divider()
                gs_c1, gs_c2 = st.columns(2)
                with gs_c1:
                    top_n_gs = st.slider(t("top_n_paths", lang), 5, 50, 10, key="gs_top_n")
                with gs_c2:
                    pt_gs = st.selectbox(t("plot_type", lang), [t("bar_plot", lang), t("dot_plot", lang)], key="gs_pt")
                
                df = st.session_state["gsea_results"].head(top_n_gs)
                if pt_gs == t("bar_plot", lang):
                    fig = plot_gsea_bar_plotly(df, f"Top {top_n_gs} GSEA Pathways (Bar)", plotly_template, sel_font, fig_font_sz)
                else:
                    fig = plot_gsea_dot_plotly(df, f"Top {top_n_gs} GSEA Pathways (Dot)", plotly_template, sel_font, fig_font_sz)
                
                st.plotly_chart(fig, width="stretch", config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"gsea_plot.{img_format}", key="dl_gsea_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(ui("### GSEA Data Table (Top {count})", lang, count=top_n_gs))
                st.dataframe(df, width="stretch")
                
        with nt2:
            st.subheader(t("tab_string", lang))
            _string_flavor_ids = ["confidence", "evidence", "actions"]
            _string_flavor_labels = {
                "confidence": ui("Confidence", lang),
                "evidence": ui("Evidence", lang),
                "actions": ui("Actions", lang),
            }
            string_flavor = st.radio(
                ui("Network Flavor", lang), _string_flavor_ids,
                format_func=_string_flavor_labels.get, horizontal=True,
                help=ui("Confidence uses line width; Evidence colors links by evidence type.", lang)
            )
            with st.expander(ui("About STRING network", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
**STRING** は実験的・計算的に予測されたタンパク質間相互作用（PPI）を収録するデータベースです。  
有意DEGのうち上位最大30遺伝子を用いてインタラクションネットワークを取得・可視化します。

- エッジの太さが相互作用スコア（信頼度）を反映
- ネットワークは **STRING-db API** からリアルタイム取得（インターネット接続必須）
- 遺伝子名が HGNC シンボルであることを確認してください
""")
                else:
                    st.markdown(ui("""
**STRING** is a database of experimentally and computationally predicted protein-protein interactions (PPI). The top 30 significant DEGs are used to fetch and visualize an interaction network.

- Edge thickness reflects interaction confidence score
- The network is fetched in real time from the **STRING-db API** (internet connection required)
- Gene names must be in HGNC symbol format
""", lang))
            _string_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if _string_disabled:
                st.info("ℹ️ " + (ui("No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun.", lang, '有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。')))
            st.info(ui("Fetching a network sends the gene list to string-db.org.", lang,
                       "ネットワーク取得時に遺伝子リストを string-db.org へ送信します。"))
            if st.button(t("string_run_btn", lang), disabled=_string_disabled):
                res_deg = st.session_state["deg_results"]
                # Use top DEGs for network
                sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()
                if not sig_genes:
                    st.warning(ui("No significant genes for network construction.", lang))
                else:
                    with st.status(ui("🎩 Fetching network...", lang)):
                        _service_event = external_service_record("string-db.org", "gene list")
                        st.session_state["external_service_events"].append(_service_event)
                        img_s = get_string_network_img(sig_genes, st.session_state["sp"]["string_id"], flavor=string_flavor, event=_service_event)
                    if img_s:
                        st.image(img_s, caption="STRING Interaction Network (Top Genes)")
                        st.download_button(ui("📥 Download STRING Network Image", lang), img_s, "string_network.png", key="dl_string_btn")
                    else:
                        st.error(ui("Could not fetch STRING network. Check internet or species ID.", lang))
            st.warning(t("string_tip", lang))
        with nt3:
            st.subheader(ui("TF Activity Estimation", lang))

            with st.expander(ui("About TF Activity Estimation", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
転写因子（TF）の活性を遺伝子発現データから推定します。TFそのものの発現量ではなく、標的遺伝子群の発現パターンから間接的に活性を算出するため、発現変動が小さいTFも捕えられます。

**CollecTRI** は実験的に検証されたTF-標的遺伝子の相互作用を収録したデータベースです。エビデンスの質が高い一方、収録TF数は少ない傾向があります。

**DoRothEA** はconfidence levelによってフィルタリングできるデータベースです。Aが最も厳格で実験的証拠に基づき、B→C→Dの順に予測ベースの低信頼性インタラクションが含まれます。なお、マウスデータに対するDoRothEAのカバレッジはCollecTRIと比較して限定的です。結果のTF数が少ない場合はConfidence levelをABCまで幅げることを検討してください。

**ULM（univariate linear model）** は各TFを独立に評価し、高速で解釈しやすい結果が得られます。  
**MLM（multivariate linear model）** は全TFを同時にモデル化し、TF間の共線性を考慮したより保守的な推定が可能です。

⚠️ **注意**: 出力されるのは各サンプルの**相対的な活性スコア**であり、群間の統計的有意差（p値）ではありません。群間の差を統計的に検定したい場合は、出力CSVのスコアを用いて別途t検定・Mann-Whitney検定などを実施してください。
""")
                else:
                    st.markdown(ui("""
Estimates transcription factor (TF) activity from gene expression data. Rather than using TF expression levels directly, activity is inferred from the expression patterns of target gene sets — enabling detection of TFs with small expression changes.

**CollecTRI** contains experimentally validated TF–target interactions. It offers high-confidence evidence but covers fewer TFs.

**DoRothEA** supports confidence-level filtering. Level A is the most stringent (experimental evidence only); B, C, and D progressively include lower-confidence, prediction-based interactions. Note: DoRothEA coverage for mouse data is limited compared to CollecTRI. If few TFs are returned, consider expanding the confidence level to ABC.

**ULM (univariate linear model)** evaluates each TF independently — fast and easy to interpret.  
**MLM (multivariate linear model)** models all TFs simultaneously, accounting for collinearity and producing more conservative estimates.

⚠️ **Note**: The output represents **relative activity scores** per sample, not statistical significance between groups (no p-values). To statistically test differences between groups, export the score CSV and perform t-tests or Mann-Whitney tests separately.
""", lang))

            # — Controls
            tf_c1, tf_c2, tf_c3 = st.columns(3)
            with tf_c1:
                method_sel     = st.selectbox(ui("Method", lang), ["ULM", "MLM"], key="tf_method")
            with tf_c2:
                dorothea_level_sel = st.selectbox(ui("DoRothEA confidence", lang), ["A", "AB", "ABC", "ABCD"], key="tf_dorothea_level")
            with tf_c3:
                min_targets    = st.slider(ui("Min. targets per TF", lang), 5, 30, 10, key="tf_min_n")
            
            if st.session_state.get("norm_method") in ["CPM", "TPM"]:
                st.caption(
                    ui("ℹ️ CPM/TPM data will be log(x+1) transformed before TF activity "
                    "estimation, as decoupleR requires log-scale input.", lang)
                )

            _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
            _tf_disabled = _deg_res_ctrl is None
            if _n_total_samples < 6:
                st.warning("⚠️ " + (ui('Only {value_0} samples. TF Activity estimation recommends 6+ samples for reliable results.', lang, 'サンプル数が{value_0}件です。TF Activity推定は6サンプル以上を推奨します（結果が不安定になる場合があります）。', value_0=_n_total_samples)))
            if _tf_disabled:
                st.warning("⚠️ " + (ui('Please run DEG analysis first.', lang, '先にDEG解析を実行してください。')))
            if st.button(ui("Estimate TF Activity", lang), type="primary", disabled=_tf_disabled):
                try:
                    sp_org   = st.session_state.get("sp", {}).get("org", "hsa")
                    organism = "human" if sp_org == "hsa" else "mouse"
                    _norm_method_tf = st.session_state.get("norm_method", "log1p")
                    _raw_norm = normalize_counts(st.session_state["counts_df"], _norm_method_tf, st.session_state.get("gene_lengths"))
                    # For TF activity, ensure log scale regardless of normalization choice
                    if _norm_method_tf in ["CPM", "TPM"]:
                        norm_mat = np.log1p(_raw_norm).T
                    else:
                        norm_mat = _raw_norm.T  # log1p and VST are already log-scale

                    with st.status(ui("🎩 Estimating TF activity...", lang), expanded=True) as status:
                        # ── decoupler v2.x API ──────────────────────────────
                        st.write(ui("⏳ Step 1/3: Loading CollecTRI network...", lang))
                        net_collectri = load_collectri_network(organism)
                        
                        st.write(ui("⏳ Step 2/3: Loading DoRothEA network...", lang))
                        levels        = list(dorothea_level_sel)  # e.g. ['A','B']
                        net_dorothea  = load_dorothea_network(organism, levels=levels)

                        st.write(ui("🎩 Step 3/3: Calculating activities using {method} (this may take a few minutes)...",
                                    lang, method=method_sel))
                        acts_c, _, method_c = infer_tf_activity(
                            norm_mat, net_collectri, method_sel, min_targets
                        )
                        acts_d, _, method_d = infer_tf_activity(
                            norm_mat, net_dorothea, method_sel, min_targets
                        )
                        fallback_resources = [
                            name for name, actual in (("CollecTRI", method_c), ("DoRothEA", method_d))
                            if actual == "ULM fallback"
                        ]
                        if fallback_resources:
                            st.warning(ui(
                                "MLM could not uniquely estimate some TFs in {resources}; BRIM safely used ULM for those network results.",
                                lang,
                                resources=", ".join(fallback_resources),
                            ))
                        
                        status.update(label="✅ TF Activity Estimation Complete!", state="complete", expanded=False)

                    st.session_state["tf_collectri"] = acts_c
                    st.session_state["tf_dorothea"]  = acts_d

                    if acts_d.shape[1] < 5:
                        st.warning(
                            ui("DoRothEA returned fewer than 5 TFs for this dataset. "
                            "This is expected for mouse data — consider expanding the confidence level (e.g. ABC) "
                            "or using CollecTRI results as the primary reference.", lang)
                        )
                    log_analysis(
                        "TF Activity",
                        f"Requested method: {method_sel}, CollecTRI method: {method_c}, "
                        f"DoRothEA method: {method_d}, DoRothEA level: {dorothea_level_sel}, "
                        f"organism: {organism}",
                    )
                    st.rerun()
                except ImportError:
                    st.error(ui("`decoupler` is not installed. Run: `pip install decoupler`", lang))
                except Exception as e:
                    st.error(ui("TF activity error: {error}", lang, error=e))

            # — Results
            if st.session_state["tf_collectri"] is not None:
                acts_c = st.session_state["tf_collectri"]
                acts_d = st.session_state["tf_dorothea"]
                res_c_tab, res_d_tab, res_cons_tab = st.tabs([ui("CollecTRI", lang), ui("DoRothEA", lang), ui("Consensus", lang)])

                # Heatmap helper
                def _tf_heatmap(acts, title_str):
                    top_tfs = acts.abs().mean(axis=0).nlargest(20).index.tolist()
                    fig_tf  = px.imshow(
                        acts[top_tfs].T,
                        aspect="auto",
                        color_continuous_scale=st.session_state.get("hm_cmap", "RdBu_r"),
                        color_continuous_midpoint=0,
                        template=plotly_template,
                        title=title_str,
                        labels={"x": "Sample", "y": "TF", "color": "Activity score"},
                    )
                    fig_tf.update_layout(font=dict(family=sel_font, size=fig_font_sz), width=fig_width, height=max(fig_height, 500))
                    return fig_tf

                with res_c_tab:
                    st.caption(ui("{count} TFs estimated (CollecTRI)", lang, count=acts_c.shape[1]))
                    fig_c_tf = _tf_heatmap(acts_c, "TF Activity — CollecTRI (top 20 by mean |activity|)")
                    st.plotly_chart(fig_c_tf, width="stretch", config=plotly_config)
                    img_c_tf = get_img_bytes(fig_c_tf, img_format, img_dpi)
                    if img_c_tf:
                        st.download_button(ui("Download plot", lang), img_c_tf, f"tf_collectri.{img_format}", key="dl_ctr_btn")
                    st.download_button(ui("Download CSV (CollecTRI)", lang),
                                       acts_c.to_csv(), "tf_collectri.csv", "text/csv", key="dl_ctr_csv")

                with res_d_tab:
                    st.caption(ui("{count} TFs estimated (DoRothEA)", lang, count=acts_d.shape[1]))
                    if acts_d.shape[1] < 5:
                        st.warning(ui("Too few TFs to display a meaningful heatmap. Try expanding the DoRothEA confidence level.", lang))
                    else:
                        fig_d_tf = _tf_heatmap(acts_d, "TF Activity — DoRothEA (top 20 by mean |activity|)")
                        st.plotly_chart(fig_d_tf, width="stretch", config=plotly_config)
                        img_d_tf = get_img_bytes(fig_d_tf, img_format, img_dpi)
                        if img_d_tf:
                            st.download_button(ui("Download plot", lang), img_d_tf, f"tf_dorothea.{img_format}", key="dl_dor_btn")
                    st.download_button(ui("Download CSV (DoRothEA)", lang),
                                       acts_d.to_csv(), "tf_dorothea.csv", "text/csv", key="dl_dor_csv")

                with res_cons_tab:
                    shared_tfs = list(set(acts_c.columns) & set(acts_d.columns))
                    if len(shared_tfs) < 3:
                        st.warning(
                            ui("Not enough shared TFs between CollecTRI and DoRothEA to generate a consensus plot. "
                            "This is common with mouse data and strict confidence levels.", lang)
                        )
                    else:
                        mean_c = acts_c[shared_tfs].mean(axis=0).rename("CollecTRI")
                        mean_d = acts_d[shared_tfs].mean(axis=0).rename("DoRothEA")
                        cons_df = pd.concat([mean_c, mean_d], axis=1).reset_index().rename(columns={"index": "TF"})

                        fig_scatter = px.scatter(
                            cons_df, x="CollecTRI", y="DoRothEA", hover_name="TF",
                            template=plotly_template,
                            title="Consensus: mean TF activity (CollecTRI vs DoRothEA)"
                        )
                        _rng = max(cons_df[["CollecTRI", "DoRothEA"]].abs().max().max(), 0.1)
                        fig_scatter.add_shape(type="line",
                            x0=-_rng, y0=-_rng, x1=_rng, y1=_rng,
                            line=dict(dash="dash", color="gray", width=1))
                        fig_scatter.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                        st.plotly_chart(fig_scatter, width="stretch", config=plotly_config)

                        top_shared = cons_df.set_index("TF").abs().mean(axis=1).nlargest(20).index.tolist()
                        hm_data    = pd.concat([
                            acts_c[top_shared].T.rename(columns=lambda s: f"CTR|{s}"),
                            acts_d[top_shared].T.rename(columns=lambda s: f"DOR|{s}")
                        ], axis=1)
                        fig_cons_hm = px.imshow(
                            hm_data,
                            aspect="auto",
                            color_continuous_scale="RdBu_r",
                            color_continuous_midpoint=0,
                            template=plotly_template,
                            title="Shared TF activity — CollecTRI (CTR|) vs DoRothEA (DOR|)"
                        )
                        fig_cons_hm.update_layout(font=dict(family=sel_font, size=fig_font_sz), width=fig_width, height=max(fig_height, 500))
                        st.plotly_chart(fig_cons_hm, width="stretch", config=plotly_config)
                        img_cons = get_img_bytes(fig_cons_hm, img_format, img_dpi)
                        if img_cons:
                            st.download_button(ui("Download consensus heatmap", lang), img_cons,
                                               f"tf_consensus.{img_format}", key="dl_cons_btn")
        with nt4:
            st.subheader(ui("Cell-type Scoring / Deconvolution", lang))
            with st.expander(ui("About Cell-type Scoring", lang), expanded=False):
                if _is_jp:
                    st.markdown("""
バルクRNA-seqデータから各サンプルの細胞型シグナルを推定します。

**利用可能な参照行列:**
- **mMCP-counter**: マウスデータ専用。マーカー遺伝子の中央値スコアを使用（回帰なし）。引用: Petitprez et al., Genome Medicine (2020)
- **LM22 / カスタムCSV**: LM22（Newman et al. 2015）など検証済みの発現量行列をCSVでアップロード。Nu-SVRによる混合比推定を行います。行=遺伝子、列=細胞型の形式で用意してください。
- **外部参照（references/フォルダ）**: アプリと同じディレクトリの `references/` フォルダに置いたCSVファイルを自動的に読み込みます。

⚠️ **注意**: Nu-SVRによる推定は「バルク発現量 ＝ 各細胞型の発現量 × 細胞割合の線形和」を前提とします。この前提が成立する**検証済みの発現量行列（LM22等）**を使用してください。マーカーリストのみから作成した参照行列では数学的前提が成立しません。

本実装は統計的信頼性（permutationによるp値）を推定しません。結果の解釈には注意してください。
""")
                else:
                    st.markdown(ui("""
Estimates cell-type signals for each sample from bulk RNA-seq data.

**Available reference matrices:**
- **mMCP-counter**: Designed for mouse data. Uses median marker gene scores (no regression). Citation: Petitprez et al., Genome Medicine (2020)
- **LM22 / Custom CSV**: Upload a validated expression matrix such as LM22 (Newman et al. 2015). Nu-SVR is used to estimate cell-type fractions. Format: rows = genes, columns = cell types.
- **External references (references/ folder)**: CSV files placed in the `references/` directory alongside the app are automatically loaded.

⚠️ **Important**: Nu-SVR assumes a linear mixture model: bulk expression = Σ (cell-type expression × fraction). Only use **validated expression matrices (e.g. LM22)** that satisfy this assumption. Reference matrices constructed from marker lists alone do not satisfy this mathematical prerequisite.

This implementation does not estimate statistical confidence (no permutation-based p-values).
""", lang))

            # — Controls
            _external_ref_options = [f"📁 {k}" for k in _EXTERNAL_REFS.keys()]
            _all_ref_options = (
                ["mMCP-counter (Mouse recommended 🐭)"]
                + _external_ref_options
                + ["Upload LM22 or custom CSV"]
            )
            ref_choice = st.radio(
                ui("Reference matrix", lang),
                _all_ref_options,
                key="decon_ref_choice"
            )
            normalize = st.checkbox(ui("Normalize fractions to sum to 1 per sample", lang), value=True, key="decon_norm")

            # 読み込み失敗した外部参照ファイルがあれば警告表示
            if _FAILED_REFS:
                for _fname, _ferr in _FAILED_REFS:
                    st.warning(
                        ui('⚠️ Failed to load external reference: `{value_0}.csv` — {value_1}', lang, '⚠️ 外部参照ファイルの読み込みに失敗しました: `{value_0}.csv` — {value_1}', value_0=_fname, value_1=_ferr)
                    )

            ref_df_upload = None
            if ref_choice == "Upload LM22 or custom CSV":
                up_ref = st.file_uploader(ui("Reference CSV (rows=genes, columns=cell types)", lang),
                                          type=["csv"], key="decon_ref_upload")
                if up_ref is not None:
                    ref_df_upload = pd.read_csv(up_ref, index_col=0)

            _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
            _decon_disabled = st.session_state.get("counts_df") is None
            if _n_total_samples < 6:
                st.warning("⚠️ " + (ui('Only {value_0} samples. Deconvolution recommends 6+ samples for reliable cell type estimation.', lang, 'サンプル数が{value_0}件です。デコンボリューションは6サンプル以上を推奨します（サンプル数が少ないと細胞型推定の精度が低下します）。', value_0=_n_total_samples)))
            if st.button(ui("Deconvolve", lang), type="primary", disabled=_decon_disabled):
                try:
                    from sklearn.svm import NuSVR

                    # Build or load reference DataFrame
                    ref_dict = None
                    is_builtin = False
                    if ref_choice == "mMCP-counter (Mouse recommended 🐭)":
                        # mMCP-counter: マーカー遺伝子の中央値スコアで計算
                        st.info(ui("🐭 mMCP-counter is designed for mouse data. Gene names must be in 'Cd8a' format. Output is abundance score, not fraction. Citation: Petitprez et al., Genome Medicine (2020)", lang, '🐭 mMCP-counter はマウスデータ専用です。遺伝子名が Cd8a 形式（先頭大文字）である必要があります。出力は割合ではなく存在量スコアです。引用: Petitprez et al., Genome Medicine (2020)'))
                        counts_decon = st.session_state["counts_df"].copy()
                        lib_sizes = counts_decon.sum(axis=0)
                        expr_data = np.log1p(counts_decon.div(lib_sizes, axis=1) * 1e6)
                        scores = {}
                        for cell_type, markers in MMCP_COUNTER_MARKERS.items():
                            available = [g for g in markers if g in expr_data.index]
                            if len(available) >= 2:
                                scores[cell_type] = expr_data.loc[available].median(axis=0)
                            else:
                                scores[cell_type] = np.nan
                        ciber_df = pd.DataFrame(scores, index=counts_decon.columns)
                        st.session_state["ciber_results"] = ciber_df
                        log_analysis("Deconvolution", "Reference: mMCP-counter (mouse)")
                        st.rerun()

                    elif ref_choice.startswith("📁 "):
                        _ext_key = ref_choice[2:].strip()
                        if _ext_key in _EXTERNAL_REFS:
                            ref_df_used = _EXTERNAL_REFS[_ext_key].copy()
                            ref_df_used.index = ref_df_used.index.str.strip()
                            is_builtin = False
                            st.info(ui("📁 External reference loaded: {name} ({genes} genes × {cell_types} cell types)",
                                       lang, name=_ext_key, genes=ref_df_used.shape[0],
                                       cell_types=ref_df_used.shape[1]))
                        else:
                            st.error(ui("Reference file not found: {name}", lang, name=_ext_key))
                            st.stop()


                    else:
                        if ref_df_upload is None:
                            st.error(ui("Please upload a reference CSV first.", lang))
                            st.stop()
                        ref_df_upload.index = ref_df_upload.index.str.strip()
                        ref_df_used = ref_df_upload.astype(float)


                                        
                    # 堅牢な共通遺伝子抽出
                    counts_decon = st.session_state["counts_df"].copy()
                    counts_decon.index = counts_decon.index.str.strip()
                    ref_df_used.index = ref_df_used.index.str.strip()
                    
                    common_genes = counts_decon.index.intersection(ref_df_used.index)
                    
                    if len(common_genes) < 5:
                        st.error("⛔ " + (ui("Only {value_0} genes match the reference. Ensure gene names are in 'CD8A' format for human or 'Cd8a' for mouse.", lang, 'リファレンスと一致する遺伝子が {value_0} 件しかありません。マウスデータなら「Cd8a」、ヒトデータなら「CD8A」形式の遺伝子名になっているか確認してください。', value_0=len(common_genes))))
                    else:
                        X = ref_df_used.loc[common_genes].values.astype(float)
                        results = {}

                        # Normalize to log-CPM for deconvolution input to reduce library size bias
                        lib_sizes = counts_decon.sum(axis=0)
                        counts_norm_decon = counts_decon.div(lib_sizes, axis=1) * 1e6

                        with st.status(ui("🎩 Running deconvolution ({count} samples)...", lang,
                                          count=len(counts_decon.columns))):
                            for sample in counts_decon.columns:
                                y = counts_norm_decon.loc[common_genes, sample].values.astype(float)
                                svr = NuSVR(kernel="linear", nu=0.5)
                                svr.fit(X, y)
                                coefs = np.maximum(svr.coef_[0], 0)
                                if normalize and coefs.sum() > 0:
                                    coefs = coefs / coefs.sum()
                                results[sample] = coefs

                        ciber_df = pd.DataFrame(results, index=ref_df_used.columns).T
                        st.session_state["ciber_results"] = ciber_df
                        log_analysis("Deconvolution",
                                     f"Reference: {ref_choice}, genes overlap: {len(common_genes)}, normalize: {normalize}")
                        st.rerun()

                except Exception as e:
                    st.error(ui("Deconvolution error: {error}", lang, error=e))

            # — Results
            if st.session_state["ciber_results"] is not None:
                ciber_df = st.session_state["ciber_results"]
                _is_builtin_res = st.session_state.get("decon_ref_choice", "").startswith("mMCP")
                _res_label = "Relative scores" if _is_builtin_res else "Estimated fractions"
                st.caption(ui("{label} for {samples} samples, {cell_types} cell types", lang,
                              label=_res_label, samples=ciber_df.shape[0], cell_types=ciber_df.shape[1]))

                # Attach condition info for coloring
                meta_dc = st.session_state.get("metadata")
                if meta_dc is not None:
                    ciber_plot = ciber_df.copy()
                    ciber_plot.index.name = "Sample"
                    ciber_plot = ciber_plot.reset_index().melt(id_vars="Sample",
                                                                var_name="Cell type",
                                                                value_name="Value")
                    _viz_title = "Immune cell score" if _is_builtin_res else "Cell-type abundance"
                    _y_axis_label = "Score" if _is_builtin_res else "Fraction"
                    
                    fig_decon = px.bar(
                        ciber_plot, x="Sample", y="Value", color="Cell type",
                        barmode="group" if _is_builtin_res else "stack",
                        template=plotly_template,
                        title=_viz_title,
                        labels={"Value": _y_axis_label}
                    )
                    fig_decon.update_layout(font=dict(family=sel_font, size=fig_font_sz),
                                            width=fig_width, height=fig_height,
                                            xaxis_tickangle=-40)
                    st.plotly_chart(fig_decon, width="stretch", config=plotly_config)
                    img_dc = get_img_bytes(fig_decon, img_format, img_dpi)
                    if img_dc:
                        st.download_button(ui("Download plot", lang), img_dc,
                                           f"celltype_viz.{img_format}", key="dl_dc_btn")

                _dl_label = "Download CSV (scores)" if _is_builtin_res else "Download CSV (fractions)"
                _dl_filename = "deconvolution_scores.csv" if _is_builtin_res else "deconvolution_fractions.csv"
                st.download_button(_dl_label,
                                   ciber_df.to_csv(), _dl_filename,
                                   "text/csv", key="dl_dc_csv")
                st.dataframe(
                    ciber_df,
                    column_config={col: st.column_config.NumberColumn(format="%.3f") for col in ciber_df.columns},
                    width="stretch"
                )
        with nt_corr:
            if st.session_state.get("counts_df") is None:
                if st.session_state.get("lang_display", "日本語") == "日本語":
                    st.info("💡 **データがありません**\n\nまず **Upload** タブでデータをアップロードしてください。")
                else:
                    st.info(ui("💡 **No data available**\n\nPlease upload data in the **Upload** tab first.", lang))
            else:
                st.subheader(ui('🔗 Gene Correlation Analysis', lang, '🔗 Gene Correlation Analysis'))
                with st.expander(ui("About Gene Correlation Analysis", lang), expanded=False):
                    if _is_jp:
                        st.markdown("""
2つの遺伝子の発現量の相関を散布図で可視化します。Pearson・Spearman相関係数とp値を表示します。

**推奨正規化方法:** log1p または VST
- **log1p**: 可視化や探索的解析に使いやすい簡便法です
- **VST**: 分散が安定しやすく、相関解析やPCAで有用なことがあります。適切性はサンプル数, 外れ値, バッチ効果, 研究目的に依存します
""")
                    else:
                        st.markdown(ui("""
Visualize the correlation between two genes as a scatter plot. Displays Pearson and Spearman correlation coefficients with p-values.

**Recommended normalization:** log1p or VST
- **log1p**: a simple and convenient choice for visualization and exploratory analysis
- **VST**: often useful for correlation analysis and PCA because variance is more stable, but suitability depends on sample size, outliers, batch effects, and study design
""", lang))

                _norm_corr = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"))
                _all_genes_corr = sorted(_norm_corr.index.tolist())
                _cc1, _cc2 = st.columns(2)
                _gene_a = _cc1.selectbox(ui("Gene A", lang), _all_genes_corr, key="corr_gene_a")
                _gene_b = _cc2.selectbox(ui("Gene B", lang), _all_genes_corr, index=min(1, len(_all_genes_corr)-1), key="corr_gene_b")
                _corr_method = st.radio(ui("Correlation method", lang), ["Pearson", "Spearman"], horizontal=True, key="corr_method")

                _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
                if _n_total_samples < 6:
                    st.warning("⚠️ " + (ui('Only {value_0} samples. Correlation analysis recommends 6+ samples (10+ samples are often preferable for more stable estimates). Results may be unreliable with fewer samples.', lang, 'サンプル数が{value_0}件です。相関解析は6サンプル以上を推奨します（6未満では相関係数・p値の信頼性が低下します。安定した推定には10サンプル以上が望ましいです）。', value_0=_n_total_samples)))
                if _gene_a != _gene_b and _gene_a in _norm_corr.index and _gene_b in _norm_corr.index:
                    _expr_a = _norm_corr.loc[_gene_a]
                    _expr_b = _norm_corr.loc[_gene_b]
                    _corr_df = pd.DataFrame({
                        "Sample": _norm_corr.columns,
                        _gene_a: _expr_a.values,
                        _gene_b: _expr_b.values,
                    })
                    if st.session_state.get("metadata") is not None:
                        _corr_df["condition"] = _corr_df["Sample"].map(st.session_state["metadata"]["condition"])
                        _color_col = "condition"
                    else:
                        _color_col = None

                    if _expr_a.nunique(dropna=True) < 2 or _expr_b.nunique(dropna=True) < 2:
                        _r, _p = np.nan, np.nan
                        st.warning(ui("Correlation is undefined because at least one selected gene has constant expression.", lang))
                    elif _corr_method == "Pearson":
                        _r, _p = stats.pearsonr(_expr_a, _expr_b)
                    else:
                        _r, _p = stats.spearmanr(_expr_a, _expr_b)

                    _fig_corr = px.scatter(
                        _corr_df, x=_gene_a, y=_gene_b,
                        color=_color_col,
                        hover_data=["Sample"],
                        template=plotly_template,
                        title=f"{_gene_a} vs {_gene_b}  |  {_corr_method} r = {_r:.3f}, p = {_p:.2e}"
                    )
                    _fig_corr.update_layout(font=dict(family=sel_font, size=fig_font_sz), width=fig_width, height=fig_height)
                    st.plotly_chart(_fig_corr, width="stretch", config=plotly_config)

                    _rc1, _rc2, _rc3 = st.columns(3)
                    _rc1.metric(f"{_corr_method} r", f"{_r:.4f}")
                    _rc2.metric(ui("p-value", lang), f"{_p:.2e}")
                    _rc3.metric(ui("Significant", lang), "Yes ✅" if _p < 0.05 else "No ❌")

                    _img_corr = get_img_bytes(_fig_corr, img_format, img_dpi)
                    if _img_corr:
                        st.download_button(ui("📥 Download ({format})", lang, format=img_format.upper()),
                                           _img_corr, f"correlation_{_gene_a}_{_gene_b}.{img_format}", key="dl_corr_btn")
                else:
                    st.info(ui('Please select two different genes for Gene A and Gene B.', lang, 'Gene A と Gene B に異なる遺伝子を選択してください。/ Please select two different genes.'))

# TAB 5: META-ANALYSIS
with tab_meta:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    _batch = st.session_state.get("batch_deg_results", {})
    _batch_provenance = st.session_state.get("batch_deg_provenance", {})
    _meta_block_reason = None
    if len(_batch) >= 2:
        _provenance_values = [_batch_provenance.get(name) for name in _batch]
        if any(study is None for study in _provenance_values):
            _meta_block_reason = ui("Study provenance is missing. Rerun Batch DEG in Multiple Studies mode.", lang)
        elif len(set(_provenance_values)) != len(_provenance_values):
            _meta_block_reason = ui("Meta-analysis requires one contrast per independent study. Select only one contrast from each study and rerun Batch DEG.", lang)
    
    _n_studies_meta   = len(st.session_state.get("multi_study_names", []))
    _n_contrasts_meta = len(st.session_state.get("batch_deg_results", {}))

    if _meta_block_reason is not None:
        st.error(_meta_block_reason)
    elif len(_batch) < 2:
        st.info(
            ui('💡 Please run 2 or more contrasts in the DEG tab first', lang, '💡 DEGタブで2つ以上のコントラストを一括実行してください')
        )
    else:
        _contrast_names = list(_batch.keys())
        _n_contrasts = len(_contrast_names)
        padj_t  = st.session_state.get("padj_t",  0.05)
        lfc_t   = st.session_state.get("lfc_t",   1.0)

        # ---- 2-2. LFC/padj integrated matrices ----
        _lfc_frames  = []
        _padj_frames = []
        _pval_frames = []
        for _cn, _rd in _batch.items():
            _lfc_frames.append(_rd[["log2FoldChange"]].rename(columns={"log2FoldChange": _cn}))
            _padj_frames.append(_rd[["padj"]].rename(columns={"padj": _cn}))
            _pval_frames.append(_rd[["pvalue"]].rename(columns={"pvalue": _cn}))
        _lfc_mat  = pd.concat(_lfc_frames,  axis=1, join="outer")
        _padj_mat = pd.concat(_padj_frames, axis=1, join="outer")
        _pval_mat = pd.concat(_pval_frames, axis=1, join="outer")

        # ---- 2-2.1 Calculate Meta-p (Fisher's method) ----
        def _calc_meta_p(row):
            pvals = row.dropna()
            if len(pvals) >= 2:
                try:
                    _, p = stats.combine_pvalues(pvals, method='fisher')
                    return p
                except (ValueError, FloatingPointError):
                    return np.nan
            return np.nan
            # len(pvals)==1の場合はNaNを返す（単一スタディは統合しない）

        _n_studies = len(set(_batch_provenance[name] for name in _contrast_names))
        _min_studies = max(2, min(_n_studies, 2))
        
        # 最低_min_studies以上のスタディで検出された遺伝子のみ統合
        _detected_in_n = _pval_mat.notna().sum(axis=1)
        # LFC方向の一貫性チェック
        # 各遺伝子について全スタディでLFCの符号が一致しているか確認
        def _check_direction_consistency(row):
            lfc_vals = _lfc_mat.loc[row.name].dropna()
            if len(lfc_vals) < 2:
                return True  # 1スタディなら一致判定不要
            signs = np.sign(lfc_vals)
            return (signs == signs.iloc[0]).all()

        _direction_consistent = _pval_mat.apply(_check_direction_consistency, axis=1)

        _integrated_p = _pval_mat.apply(_calc_meta_p, axis=1)

        # BH補正は全遺伝子に対して先に適用する（FDR保証のため）
        # NaN（Fisher法が計算できなかった遺伝子）は1.0として補正に含める
        _integrated_p_filled = _integrated_p.fillna(1.0)
        _, _meta_padj_all, _, _ = multipletests(_integrated_p_filled, method='fdr_bh')
        _meta_padj_series = pd.Series(_meta_padj_all, index=_integrated_p.index)

        # 補正後に生物学的フィルタを適用
        # 方向が一致しない遺伝子・検出スタディ数が不足する遺伝子はNaNに設定
        _meta_padj_series[~_direction_consistent] = np.nan
        _meta_padj_series[_detected_in_n < _min_studies] = np.nan

        _padj_mat['meta_padj'] = _meta_padj_series.values

        # ---- 2-3. Filtering UI ----
        st.header(ui('🔬 Meta-Analysis', lang, '🔬 メタ解析（複数コントラスト統合）'))
        st.warning(
            ui("⚠️ **Statistical note**: Fisher's method for combining p-values requires that each contrast is **statistically independent**. Combining contrasts that share a common control group within the same study (e.g., Drug A vs Control and Drug B vs Control) violates this independence assumption and inflates false positives. **Use only across independent studies.**", lang, '⚠️ **統計的注意**: Fisher法によるp値統合は、各コントラストのp値が**統計的に独立**であることを前提とします。同一Study内で共通のコントロール群を持つコントラスト（例：Drug A vs Control と Drug B vs Control）を統合すると、独立性の前提が破綻し偽陽性が増加します。**独立した別々のStudy間でのみ使用してください。**')
        )
        _mc1, _mc2, _mc3 = st.columns(3)
        with _mc1:
            _meta_dir_ids = ["up", "down", "both"]
            _meta_dir_labels = {
                "up": ui("Up", lang),
                "down": ui("Down", lang),
                "both": ui("Both", lang),
            }
            _meta_dir = st.radio(
                ui('Direction', lang, '方向フィルター'),
                _meta_dir_ids, format_func=_meta_dir_labels.get,
                index=2, horizontal=True, key="meta_dir"
            )
        with _mc2:
            _meta_min_n = st.slider(
                ui('Min contrasts (significant)', lang, '有意コントラスト数（最少）'),
                1, _n_contrasts, min(2, _n_contrasts), key="meta_min_n"
            )
        with _mc3:
            if _n_studies == 2:
                _min_studies_ui = 2
                st.metric(ui('Min. studies detected', lang, '最低検出スタディ数'), 2)
            else:
                _min_studies_ui = st.slider(
                    ui('Min. studies detected', lang, '最低検出スタディ数'),
                    min_value=2,
                    max_value=_n_studies,
                    value=min(2, _n_studies),
                    key="meta_min_studies"
                )
            _meta_padj_t = st.number_input(
                ui('Meta padj threshold', lang, '統合p値の閾値'),
                0.0, 1.0, 0.05, 0.005, key="meta_padj_t"
            )
            _meta_run = st.button(ui('Plot', lang, 'プロット'), key="meta_plot_btn", type="primary", use_container_width=True)
        
        # BH補正は常に全遺伝子で先に行う（FDR保証のため）
        # NaN（Fisher法が計算できなかった遺伝子）は1.0として補正に含める
        _integrated_p_ui_filled = _integrated_p.fillna(1.0)
        _, _meta_padj_all_ui, _, _ = multipletests(_integrated_p_ui_filled, method='fdr_bh')
        _meta_padj_ui_series = pd.Series(_meta_padj_all_ui, index=_integrated_p.index)

        # 補正後に方向不一致・min_studies_uiの条件でNaN化（フィルタは補正後）
        _meta_padj_ui_series[~_direction_consistent] = np.nan
        _meta_padj_ui_series[_detected_in_n < _min_studies_ui] = np.nan

        _padj_mat['meta_padj'] = _meta_padj_ui_series.values

        # Download full LFC matrix (unfiltered) containing meta_padj
        _export_meta = pd.concat([_lfc_mat, _padj_mat[['meta_padj']]], axis=1)
        st.session_state["lfc_meta_matrix"] = _export_meta.copy()
        st.download_button(
            ui('📥 Download LFC Matrix CSV', lang, '📥 LFC統合マトリクス CSV'),
            _export_meta.to_csv(),
            "lfc_integrated_matrix.csv",
            key="meta_dl_lfc"
        )

        if _meta_run:
            # ---- 2-4. Filter ----
            _sig_bool = pd.DataFrame(index=_padj_mat.index)
            for _cn in _contrast_names:
                _p  = _padj_mat[_cn].fillna(1.0)
                _lf = _lfc_mat[_cn].fillna(0.0)
                if _meta_dir == "up":
                    _sig_bool[_cn] = (_p < padj_t) & (_lf > lfc_t)
                elif _meta_dir == "down":
                    _sig_bool[_cn] = (_p < padj_t) & (_lf < -lfc_t)
                else:
                    _sig_bool[_cn] = (_p < padj_t) & (_lf.abs() > lfc_t)

            _n_sig = _sig_bool.sum(axis=1)
            # 統合p値の条件を追加
            _is_meta_sig = _padj_mat['meta_padj'].fillna(1.0) < _meta_padj_t
            _keep  = _n_sig[(_n_sig >= _meta_min_n) & _is_meta_sig].index

            if len(_keep) == 0:
                st.warning(
                    ui('No genes met the specified criteria.', lang, '指定された条件で有意な遙伝子が見つかりませんでした。')
                )
            else:
                _lfc_filt = _lfc_mat.loc[_keep]

                # ---- 2-5. LFC Heatmap ----
                _heat_h = max(400, len(_keep) * 18)
                _fig_heat = go.Figure(go.Heatmap(
                    z=_lfc_filt.values.tolist(),
                    x=_lfc_filt.columns.tolist(),
                    y=_lfc_filt.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    connectgaps=False,
                ))
                _fig_heat.update_layout(
                    title=(ui('🔥 LFC Integrated Heatmap', lang, '🔥 LFC統合ヒートマップ')),
                    height=_heat_h,
                    yaxis={"tickfont": {"size": 9}},
                    xaxis_title="Contrast",
                    yaxis_title="Gene",
                    font=dict(family=sel_font, size=fig_font_sz),
                    template=plotly_template,
                )
                st.plotly_chart(_fig_heat, width="stretch")

                # ---- 2-6. Venn / UpSet (自動切替) ----
                st.divider()
                _deg_sets = {
                    _cn: set(_sig_bool[_sig_bool[_cn]].index)
                    for _cn in _contrast_names
                }
                _dir_label = _meta_dir_labels[_meta_dir]

                if _n_contrasts <= 3:
                    _plot_mode_meta = "Venn"
                    st.caption("💡 " + (ui('≤3 contrasts → showing Venn diagram.', lang, 'コントラスト数が3以下のためVenn図を表示します。')))
                else:
                    _plot_mode_meta = "UpSet"
                    st.caption("💡 " + (ui('≥4 contrasts → showing UpSet plot.', lang, 'コントラスト数が4以上のためUpSet図を表示します。')))

                if not _deg_sets or all(len(v) == 0 for v in _deg_sets.values()):
                    st.warning(ui('No DEGs found with current thresholds.', lang, '現在の閾値ではDEGが見つかりませんでした。')) # A-11
                else:
                    if _plot_mode_meta == "Venn":
                        # Venn
                        try:
                            import matplotlib_venn
                            _fig_v, _ax_v = plt.subplots(figsize=(7, 5))
                            _sets_list = [_deg_sets[c] for c in _contrast_names]
                            if _n_contrasts == 2:
                                matplotlib_venn.venn2(_sets_list, set_labels=_contrast_names, ax=_ax_v)
                            else:
                                matplotlib_venn.venn3(_sets_list, set_labels=_contrast_names, ax=_ax_v)
                            _ax_v.set_title(f"Venn — {_dir_label}")
                            st.pyplot(_fig_v)
                            plt.close(_fig_v)
                        except ImportError:
                            st.error("pip install matplotlib-venn が必要です / Please run: pip install matplotlib-venn")
                    else:
                        # UpSet
                        try:
                            from upsetplot import from_memberships, UpSet
                            _all_g = sorted(set.union(*_deg_sets.values()))
                            _membs = [
                                [_cn for _cn in _contrast_names if _g in _deg_sets[_cn]]
                                for _g in _all_g
                            ]
                            _membs = [m for m in _membs if m]
                            if _membs:
                                _ud = from_memberships(_membs)
                                _fig_up, _ = plt.subplots(figsize=(14, 6))
                                UpSet(_ud, show_counts=True).plot(fig=_fig_up)
                                plt.suptitle(f"UpSet — {_dir_label}", y=1.02)
                                st.pyplot(_fig_up)
                                plt.close(_fig_up)
                        except ImportError:
                            st.error("pip install upsetplot が必要です / Please run: pip install upsetplot")

# TAB 6: EXPORT
with tab_export:
    if st.session_state["deg_results"] is not None:
        st.subheader(ui("📦 Package Export", lang))
        export_files = collect_all_results()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f, d in export_files.items():
                z.writestr(f, d)
        st.download_button(
            ui("📦 Download Results ZIP", lang),
            buf.getvalue(),
            "results.zip",
            "application/zip",
            key="dl_zip_btn"
        ) # A-16
        
        st.divider()
        st.subheader(ui("📝 Reproducibility", lang))
        st.download_button(t("dl_report", lang), export_files["Provenance/manifest.json"],
                           "manifest.json", "application/json")
        st.download_button(ui("Download provenance (Markdown)", lang, "Provenanceをダウンロード (Markdown)"),
                           export_files["Provenance/manifest.md"], "manifest.md", "text/markdown")
    else:
        _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
        if _is_jp:
            st.info("💡 **出力するデータがありません**\n\nまずは **DEG** タブで解析を実行して結果を生成してください。")
        else:
            st.info(ui("💡 **No data to export**\n\nPlease run analysis in the **DEG** tab to generate results first.", lang))

# TAB 6: INFO
with tab_info:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    st.header(t("tab_info", lang))
    st.divider()
    st.subheader(ui("Environment & Versions", lang))
    
    # Collect package versions
    def get_pkg_version(pkg_name):
        try:
            import importlib.metadata
            return importlib.metadata.version(pkg_name)
        except Exception:
            return "N/A"

    env_data = {
        "App Version": APP_VERSION,
        "Python": sys.version.split()[0],
        "OS": platform.system() + " " + platform.release(),
        "Streamlit": get_pkg_version("streamlit"),
        "PyDESeq2": get_pkg_version("pydeseq2"),
        "Plotly": get_pkg_version("plotly"),
        "Pandas": get_pkg_version("pandas"),
        "NumPy": get_pkg_version("numpy"),
        "scikit-learn": get_pkg_version("scikit-learn"),
        "GSEApy": get_pkg_version("gseapy"),
        "kaleido": get_pkg_version("kaleido"),
    }
    
    ver_cols = st.columns(3)
    for i, (k, v) in enumerate(env_data.items()):
        ver_cols[i % 3].metric(k, v)


    st.divider()
    st.info(t("notebook_desc", lang))
    if not st.session_state["analysis_log"]:
        st.write(ui('No analysis history yet. Please run an analysis.', lang, '履歴がありません。解析を実行してください。')) # BUG FIX: バグ⑧
    else:
        for entry in st.session_state["analysis_log"]:
            with st.expander(f"[{entry['time']}] {entry['action']}", expanded=True):
                st.markdown(entry["details"].replace("\n", "  \n"))
        
        all_res = collect_all_results()
        if "Analysis_Notebook.md" in all_res:
            st.download_button(t("dl_notebook", lang), all_res["Analysis_Notebook.md"], "Analysis_Notebook.md")

    # ─── References ─────────────────────────────────────────────
    st.divider()
    st.subheader(ui("📚 References", lang))
    if _is_jp:
        st.write("本アプリの解析は以下のツールとデータベースに支えられています。論文発表の際は、使用した機能に応じて該当する文献を引用してください。")
    else:
        st.write(ui("This application relies on the following tools and databases. Please cite the respective publications based on the features you used.", lang))
    with st.expander(ui("Show all references", lang), expanded=False):
        st.markdown(ui("""
#### 1. Differential Expression Analysis (DEG)
- **PyDESeq2**: Muzellec, L. et al., *Bioinformatics* (2023). [DOI: 10.1093/bioinformatics/btad547](https://doi.org/10.1093/bioinformatics/btad547)
- **DESeq2**: Love, M. I. et al., *Genome Biology* (2014). [DOI: 10.1186/s13059-014-0550-8](https://doi.org/10.1186/s13059-014-0550-8)

#### 2. Pathway & Gene Set Enrichment Analysis (KEGG, GO, GSEA)
- **GSEApy**: Fang, Z. et al., *Bioinformatics* (2022). [DOI: 10.1093/bioinformatics/btac757](https://doi.org/10.1093/bioinformatics/btac757)
- **Enrichr**: Kuleshov, M. V. et al., *Nucleic Acids Research* (2016). [DOI: 10.1093/nar/gkw377](https://doi.org/10.1093/nar/gkw377)
- **GSEA**: Subramanian, A. et al., *PNAS* (2005). [DOI: 10.1073/pnas.0506580102](https://doi.org/10.1073/pnas.0506580102)

#### 3. Transcription Factor Activity
- **decoupleR**: Badia-i-Mompel, A. et al., *Bioinformatics Advances* (2022). [DOI: 10.1093/bioadv/vbac016](https://doi.org/10.1093/bioadv/vbac016)
- **CollecTRI**: Müller-Dott, S. et al., *Nucleic Acids Research* (2023). [DOI: 10.1093/nar/gkad841](https://doi.org/10.1093/nar/gkad841)
- **DoRothEA**: Garcia-Alonso, L. et al., *Genome Research* (2019). [DOI: 10.1101/gr.240663.118](https://doi.org/10.1101/gr.240663.118)

#### 4. Immune Deconvolution
- **CIBERSORT / Nu-SVR**: Newman, A. M. et al., *Nature Methods* (2015). [DOI: 10.1038/nmeth.3337](https://doi.org/10.1038/nmeth.3337)

#### 5. Protein-Protein Interaction Network
- **STRING**: Szklarczyk, D. et al., *Nucleic Acids Research* (2023). [DOI: 10.1093/nar/gkac1000](https://doi.org/10.1093/nar/gkac1000)
""", lang))
