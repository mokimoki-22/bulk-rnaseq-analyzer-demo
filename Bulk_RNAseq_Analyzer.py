import datetime
import io
import re
import sys
import time
import platform
import zipfile
from scipy import stats

APP_VERSION = "1.1.0"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.gridspec import GridSpec

# i18n
from i18n import LANGUAGE_OPTIONS, t

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

# Built-in immune cell reference matrices for deconvolution
ICE_REFERENCE = {
    "CD8_T_cell":    ["CD8A", "CD8B", "GZMB", "PRF1", "IFNG"],
    "CD4_T_cell":    ["CD4", "IL7R", "FOXP3", "IL2RA", "CTLA4"],
    "NK_cell":       ["NCAM1", "NKG7", "GNLY", "KLRD1", "FCER1G"],
    "B_cell":        ["CD19", "MS4A1", "CD79A", "IGHM", "PAX5"],
    "Monocyte":      ["CD14", "LYZ", "S100A8", "S100A9", "CSF1R"],
    "Macrophage_M1": ["IL1B", "TNF", "CXCL10", "NOS2", "CD80"],
    "Macrophage_M2": ["MRC1", "CD163", "IL10", "ARG1", "TGFB1"],
    "Neutrophil":    ["ELANE", "MPO", "CXCR2", "S100A12", "FCGR3B"],
    "Dendritic_cell":["ITGAX", "CD1C", "CLEC9A", "FCER1A", "CCR7"],
    "Mast_cell":     ["TPSAB1", "CPA3", "KIT", "MS4A2", "HPGDS"],
}
HCA_REFERENCE = {
    "CD8_T_cell":    ["CD8A", "GZMK", "CCL5", "CXCR3", "LAG3"],
    "CD4_T_cell":    ["CD4", "CCR7", "TCF7", "LEF1", "SELL"],
    "NK_cell":       ["FCGR3A", "CX3CR1", "TYROBP", "GZMH", "KLRF1"],
    "B_cell":        ["CD79B", "BANK1", "RALGPS2", "BLK", "FCRL1"],
    "Monocyte":      ["VCAN", "FCN1", "SELL", "CD36", "CLEC12A"],
    "Macrophage_M1": ["CXCL9", "IDO1", "STAT1", "GBP1", "APOL1"],
    "Macrophage_M2": ["LYVE1", "FOLR2", "STAB1", "DAB2", "CCL18"],
    "Neutrophil":    ["CEACAM8", "PGLYRP1", "OLFM4", "LCN2", "MMP8"],
    "Dendritic_cell":["LILRA4", "CLEC4C", "IL3RA", "IRF7", "SIGLEC6"],
    "Mast_cell":     ["GATA2", "HDC", "PLCG1", "SLC18A2", "CD63"],
}
NEURAL_REFERENCE = {
    "Astrocyte":         ["AQP4", "GFAP", "ALDH1L1", "SLC1A2", "S100B"],
    "Microglia":         ["AIF1", "TMEM119", "CX3CR1", "P2RY12", "ITGAM"],
    "Oligodendrocyte":   ["MBP", "MOG", "PLP1", "SOX10", "OLIG2"],
    "Neuron_Excitatory": ["SLC17A6", "SLC17A7", "GRIN1", "GRIA1", "CAMK2A"],
    "Neuron_Inhibitory": ["GAD1", "GAD2", "SLC32A1", "PVALB", "SST"],
    "Endothelial":       ["PECAM1", "CDH5", "TIE1", "FLT1", "VWF"],
}

# Results storage
if "enr_kegg"     not in st.session_state: st.session_state["enr_kegg"]     = None
if "enr_go"       not in st.session_state: st.session_state["enr_go"]       = None
if "gsea_results" not in st.session_state: st.session_state["gsea_results"] = None
if "tf_results"   not in st.session_state: st.session_state["tf_results"]   = None
if "tf_collectri" not in st.session_state: st.session_state["tf_collectri"] = None
if "tf_dorothea"  not in st.session_state: st.session_state["tf_dorothea"]  = None
if "ciber_results" not in st.session_state: st.session_state["ciber_results"] = None
if "fig_font_sz"  not in st.session_state: st.session_state["fig_font_sz"]  = 12

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


def log_analysis(action, details=""):
    st.session_state["analysis_log"].append({"time": datetime.datetime.now().strftime("%H:%M:%S"), "action": action, "details": details})

@st.cache_data(show_spinner=False)
def get_string_network_img(gene_list, species_id, limit=30, flavor="confidence"):
    url = "https://string-db.org/api/image/network"
    params = {
        "identifiers": "\r".join(gene_list[:limit]),
        "species": species_id,
        "add_white_nodes": 1,
        "network_flavor": flavor
    }
    try:
        res = requests.post(url, data=params, timeout=30)
        return res.content if res.status_code == 200 else None
    except: return None

# ═══════════════════════════════════════════
# 3. ANALYSIS HELPERS
# ═══════════════════════════════════════════
@st.cache_data(show_spinner=False)
def run_online_mapping(id_list, species_id):
    mapped_dict = {}
    for i in range(0, len(id_list), 1000):
        chunk = id_list[i:i+1000]
        try:
            res = requests.post("https://mygene.info/v3/query", data={'q':",".join(chunk),'scopes':'ensembl.gene,entrezgene,refseq,uniprot','species':species_id,'fields':'symbol'}, timeout=30)
            if res.status_code == 200:
                for item in res.json():
                    if 'symbol' in item: mapped_dict[item.get('query')] = item['symbol']
        except: pass
        time.sleep(0.3)
    return mapped_dict



def normalize_counts(counts_df, method="log1p"):
    """Apply selected normalization to count matrix."""
    if method == "log1p":
        return np.log1p(counts_df)
    elif method == "CPM":
        lib_size = counts_df.sum(axis=0)
        return (counts_df.div(lib_size, axis=1) * 1e6)
    elif method == "TPM":
        # gene length unavailable: fallback to CPM with note
        lib_size = counts_df.sum(axis=0)
        return (counts_df.div(lib_size, axis=1) * 1e6)
    elif method == "VST":
        try:
            from pydeseq2.preprocessing import vst_fit_transform
            return pd.DataFrame(
                vst_fit_transform(counts_df.T.values),
                index=counts_df.columns,
                columns=counts_df.index
            ).T
        except Exception:
            return np.log1p(counts_df)
    return np.log1p(counts_df)

# ── 共通遺伝子リスト（Single / Multi Study 両方で使用） ──────────────
_IMMUNE_GENES = [
    "Tnf","Il6","Il1b","Il10","Il4","Il13","Ifng","Tgfb1","Ccl2","Ccl5",
    "Cxcl1","Cxcl10","Stat1","Stat3","Stat6","Nfkb1","Irf3","Irf7","Tlr4","Myd88",
    "Cd4","Cd8a","Foxp3","Gata3","Tbx21","Rorc","Il17a","Il22","Il33","Il25",
    "Tslp","Il31","Il5","Csf2","Vegfa","Mmp9","Mmp2","Timp1","Col1a1","Col3a1",
    "Fn1","Vim","Cdh1","Ocln","Cldn1","Krt1","Krt10","Filaggrin","Loricrin","Involucrin"
]
_BARRIER_GENES = [
    "Flg","Lce1a","Lce1b","Lce2a","Krt2","Krt5","Krt14","Krt16",
    "Dsg1","Dsg3","Dsp","Pkp1","Gja1","Aqp3","Smpd1","Cers3","Elovl4","Fa2h",
    "Abca12","Cldn4","Cldn7","Cldn11","Tjp1","Ocln2","Cdh2","Itgb4","Itga6",
    "Lamb3","Lamc2","Lama3"
]
_SIGNAL_GENES = [
    "Egfr","Erbb2","Fgfr1","Pdgfra","Igf1r","Insr","Met",
    "Akt1","Akt2","Pten","Mtor","Mapk1","Mapk3","Mapk8","Mapk14","Jnk1",
    "Pik3ca","Pik3r1","Kras","Hras","Nras","Braf","Raf1","Map2k1",
    "Jak1","Jak2","Tyk2","Socs1","Socs3","Ptpn11","Grb2","Sos1","Shc1",
    "Plcg1","Prkca","Prkcb","Prkcd","Calm1","Camk2a","Creb1","Jun","Fos",
    "Myc","Tp53","Rb1","Cdkn1a","Cdkn2a","Bcl2","Bax"
]
_EXTRA_GENES = [f"Gene{str(i).zfill(3)}" for i in range(1, 371)]
_ALL_GENES = _IMMUNE_GENES + _BARRIER_GENES + _SIGNAL_GENES + _EXTRA_GENES  # 計500遺伝子

@st.cache_data(show_spinner=False)
def generate_sample_data():
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
    for g in ["Flg","Loricrin","Involucrin","Krt1","Krt10","Cldn1","Ocln"]:
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
    for g in ["Flg","Loricrin","Krt1"]:
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
    for g in ["Flg","Loricrin","Involucrin","Cldn1","Ocln","Krt1","Krt10","Dsg1"]:
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

@st.cache_data(show_spinner=False)
def run_deg(counts_df, metadata, ref_condition, test_condition, n_cpus=1):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    # Filter to only the two groups being compared
    metadata = metadata[metadata["condition"].isin([ref_condition, test_condition])]
    count_matrix = counts_df.T.loc[metadata.index]
    dds = DeseqDataSet(counts=count_matrix, metadata=metadata, design="~condition", refit_cooks=(metadata["condition"].value_counts().min() >= 3), ref_level=["condition", ref_condition], n_cpus=n_cpus)
    dds.deseq2()
    stat_res = DeseqStats(dds, contrast=["condition", test_condition, ref_condition], n_cpus=n_cpus)
    stat_res.summary()
    res = stat_res.results_df.copy()
    res["padj"] = res["padj"].fillna(1.0)
    res["log2FoldChange"] = res["log2FoldChange"].fillna(0.0)
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

def plot_ma_plotly(df, padj_th, up_c, down_c, template='plotly_white', font="sans-serif", highlight_gene=None, font_size=12):
    df_plot = df.copy().reset_index()
    gene_col = df_plot.columns[0]
    df_plot['Status'] = 'NS'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] > 0), 'Status'] = 'Up'
    df_plot.loc[(df_plot['padj'] < padj_th) & (df_plot['log2FoldChange'] < 0), 'Status'] = 'Down'
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
    except:
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
    except:
        df_plot['Size'] = 10
        
    fig = px.scatter(df_plot, x='NES', y='Term', size='Size', color='FDR q-val',
                     hover_data=['FDR q-val', 'NOM p-val'],
                     title=title, template=template, color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"))
    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=font, size=font_size))
    return fig

def collect_all_results():
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
    return files

# ═══════════════════════════════════════════
# 5. PAGE SETUP
# ═══════════════════════════════════════════
st.set_page_config(page_title="BulkSeq Analyzer", page_icon="🧬", layout="wide")

# — Sidebar: Analysis Status (Quick Navigation) ───
with st.sidebar:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    st.markdown(f"### {'📊 解析ステータス' if _is_jp else '📊 Analysis Status'}")

    # ── 各ステップの完了判定 ──────────────────────────────
    _upload_mode_sb = st.session_state.get("upload_mode", "single")
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
            (s1, "データアップロード" if _is_jp else "Upload Data"),
            (s2, "群の設定" if _is_jp else "Group Assignment"),
            (s3, "DEG解析" if _is_jp else "DEG Analysis"),
            (s4, "TF / Network解析" if _is_jp else "TF / Network"),
        ]
        # 次のアクション案内
        if st.session_state.get("counts_df") is None:
            _next = "⬆️ " + ("Uploadタブでデータをアップロード" if _is_jp else "Upload your data in the Upload tab")
        elif st.session_state.get("metadata") is None:
            _next = "⬆️ " + ("Uploadタブで群を設定してください" if _is_jp else "Assign groups in the Upload tab")
        elif st.session_state.get("deg_results") is None:
            _next = "➡️ " + ("DEGタブでAnalyzeを実行" if _is_jp else "Run Analyze in the DEG tab")
        else:
            _next = "➡️ " + ("VisualizationまたはNetworkタブへ" if _is_jp else "Go to Visualization or Network tab")
    else:
        _status_items = [
            (s1, f"データアップロード（{_n_studies_sb} Study）" if _is_jp else f"Upload ({_n_studies_sb} Studies)"),
            (s2, "群・Study設定" if _is_jp else "Group / Study Assignment"),
            (s_batch, f"一括DEG解析（{_n_batch_sb}コントラスト）" if _is_jp else f"Batch DEG ({_n_batch_sb} contrasts)"),
            (s_meta, "Meta LFC統合" if _is_jp else "Meta LFC Integration"),
            (s4, "TF / Network解析" if _is_jp else "TF / Network"),
        ]
        if st.session_state.get("counts_df") is None:
            _next = "⬆️ " + ("Uploadタブでファイルをアップロード" if _is_jp else "Upload files in the Upload tab")
        elif _n_batch_sb < 2:
            _next = "➡️ " + ("DEGタブで一括実行（2コントラスト以上）" if _is_jp else "Run Batch DEG (2+ contrasts) in DEG tab")
        elif st.session_state.get("lfc_meta_matrix") is None:
            _next = "➡️ " + ("MetaタブでLFC統合を実行" if _is_jp else "Run LFC integration in Meta tab")
        else:
            _next = "✅ " + ("全ステップ完了" if _is_jp else "All steps complete")

    # ── ステータスカード描画 ─────────────────────────────
    _items_html = "".join([
        f"<div style='font-size:12px; margin-bottom:5px;'>{_s} <b>{_label}</b></div>"
        for _s, _label in _status_items
    ])
    st.markdown(f"""
    <div style='background:rgba(79,110,247,0.05); padding:10px; border-radius:8px; border:1px solid rgba(79,110,247,0.1); margin-bottom:10px;'>
        {_items_html}
    </div>
    <div style='background:rgba(79,110,247,0.08); padding:8px 10px; border-radius:6px; border-left:3px solid #4F6EF7; font-size:12px; color:#4F6EF7; margin-bottom:15px;'>
        {_next}
    </div>
    """, unsafe_allow_html=True)
    st.divider()

# — Sidebar: Language & Theme
with st.sidebar.expander("Language & Theme", expanded=True):
    _lang_opts = list(LANGUAGE_OPTIONS.keys())
    sel_lang_display = st.selectbox("", _lang_opts,
        index=_lang_opts.index(st.session_state.get("lang_display", "日本語")))
    st.session_state["lang_display"] = sel_lang_display
    lang = LANGUAGE_OPTIONS[sel_lang_display]
    theme_opts = [t("theme_light", lang), t("theme_dark", lang), "Ocean"]
    _saved = st.session_state.get("theme_choice", "")
    _idx = 2 if "Ocean" in _saved else (1 if ("Dark" in _saved or "ダーク" in _saved) else 0)
    sel_theme = st.selectbox("Theme", theme_opts, index=_idx)
    st.session_state["theme_choice"] = sel_theme

lang = LANGUAGE_OPTIONS[st.session_state.get("lang_display", "日本語")]
sel_theme = st.session_state.get("theme_choice", "Light") # BUG FIX: バグ⑤
is_dark  = "Dark" in sel_theme or "ダーク" in sel_theme
is_ocean = "Ocean" in sel_theme
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

# — Sidebar: Figure settings
with st.sidebar.expander("Figure Settings", expanded=True):
    # 選択された言語のフォントをリストの先頭に追加して重複排除
    dynamic_fonts = list(dict.fromkeys([app_font_name] + FONTS))
    _saved_font = st.session_state.get("selected_font", app_font_name)
    if _saved_font not in dynamic_fonts: 
        _saved_font = app_font_name
    
    sel_font   = st.selectbox("Font", dynamic_fonts, index=dynamic_fonts.index(_saved_font))
    st.session_state["selected_font"] = sel_font
    img_format = st.selectbox("Export format", ["png", "pdf", "svg"], index=0)
    img_dpi    = st.selectbox("DPI", [300, 600], index=0)
    up_color   = st.color_picker("Up color",   st.session_state.get("up_color",   "#E64B35"))
    down_color = st.color_picker("Down color", st.session_state.get("down_color", "#4DBBD5"))
    st.session_state["up_color"]   = up_color
    st.session_state["down_color"] = down_color
    fig_width   = st.slider("Width (px)",  400, 1600, st.session_state.get("fig_width",  800), 50)
    fig_height  = st.slider("Height (px)", 300, 1200, st.session_state.get("fig_height", 500), 50)
    st.session_state["fig_width"]  = fig_width
    st.session_state["fig_height"] = fig_height
    fig_font_sz = st.slider(
        "Font size (pt)", 8, 28,
        st.session_state.get("fig_font_sz", 12), 1,
        key="fig_font_sz" # BUG FIX: バグ⑥
    )
    # 各群の個別色設定
    if st.session_state.get("conditions"):
        with st.sidebar.expander("Group Colors", expanded=False):
            cond_colors = {}
            default_palette = sns.color_palette("husl", len(st.session_state["conditions"])).as_hex()
            for i, cond in enumerate(st.session_state["conditions"]):
                key = f"color_{cond}"
                color = st.color_picker(f"Color for {cond}", default_palette[i], key=key)
                cond_colors[cond] = color
            st.session_state["custom_cond_colors"] = cond_colors
    with st.sidebar.expander("Analysis Plot Colors", expanded=False):
        enr_cmap = st.selectbox("Enrichment Scale (KEGG/GO/GSEA)", ["Viridis_r", "Plasma_r", "Magma_r", "Inferno_r", "Cividis_r"], index=0)
        hm_cmap = st.selectbox("Heatmap Scale (TF/DEG Heatmap)", ["RdBu_r", "Spectral_r", "Coolwarm", "RdYlBu_r"], index=0)
        st.session_state["enr_cmap"] = enr_cmap
        st.session_state["hm_cmap"] = hm_cmap

# — Sidebar: Thresholds
with st.sidebar.expander("Thresholds", expanded=True):
    if "lfc_t"  not in st.session_state: st.session_state["lfc_t"]  = 1.0
    if "padj_t" not in st.session_state: st.session_state["padj_t"] = 0.05
    lfc_t  = st.slider(t("logfc_threshold", lang), 0.0, 5.0,   st.session_state["lfc_t"],  0.25)
    st.caption("LFC threshold: " + ("log2FoldChange の絶対値の閾値です。1.0 = 2倍変動、2.0 = 4倍変動に相当します。迷ったら 1.0 から始めてください。" if _is_jp else "Absolute log2FoldChange threshold. 1.0 = 2-fold, 2.0 = 4-fold change. Start with 1.0 if unsure."))
    padj_t = st.slider(t("pval_threshold",  lang), 0.001, 0.1, st.session_state["padj_t"], 0.005)
    st.caption("padj threshold: " + ("多重検定補正済みp値の閾値です。0.05（5%）が一般的です。DEGが少なすぎる場合は 0.1 に緩めてみてください。" if _is_jp else "Adjusted p-value threshold. 0.05 (5%) is standard. Try 0.1 if too few DEGs are found."))
    st.session_state["lfc_t"]  = lfc_t
    st.session_state["padj_t"] = padj_t
    st.session_state["deg_t"]  = (lfc_t, padj_t)

# — Sidebar: Low-count Filtering
with st.sidebar.expander("Low-count Filtering", expanded=False):
    do_filter = st.checkbox("Enable filtering", value=False, key="filter_enable")
    min_count = st.number_input("Minimum count threshold", 0, 1000, 10, key="filter_min_count")
    min_samples = st.number_input("Minimum samples expressing gene", 1, 100, 2, key="filter_min_samples")

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

st.markdown(f"""
<div style='display:flex; align-items:center; gap:14px; margin-bottom:4px;'>
  <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="24" cy="32" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="18" cy="28" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="30" cy="29" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="22" cy="24" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="26" cy="26" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="14" cy="30" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="34" cy="31" r="2" fill="#94a3b8" opacity="0.4"/>
    <circle cx="10" cy="10" r="3" fill="#E64B35"/>
    <circle cx="38" cy="8" r="3.5" fill="#E64B35"/>
    <circle cx="8" cy="16" r="2.5" fill="#E64B35" opacity="0.8"/>
    <circle cx="40" cy="14" r="2.5" fill="#E64B35" opacity="0.8"/>
    <circle cx="12" cy="12" r="2.5" fill="#4DBBD5" opacity="0.9"/>
    <circle cx="36" cy="11" r="2" fill="#4DBBD5" opacity="0.8"/>
    <line x1="4" y1="38" x2="44" y2="38" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>
    <line x1="24" y1="4" x2="24" y2="38" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-dasharray="3 2"/>
  </svg>
  <div>
    <div style='font-size:1.6em; font-weight:700; line-height:1.2;'>Bulk RNA-seq Analyzer <span style='font-size:0.4em; color:gray; font-weight:400;'>v{APP_VERSION}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)
st.caption(t('app_subtitle', lang))

# — Persistent threshold display
_lfc_cur  = st.session_state.get("lfc_t",  1.0)
_padj_cur = st.session_state.get("padj_t", 0.05)
st.caption(f"Active thresholds: LFC > {_lfc_cur} | padj < {_padj_cur}")

# サンプルデータ使用中のバッジ表示
if st.session_state.get("is_sample_data", False):
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    msg = "🧪 **サンプルデータを使用中 (Mouse, 9 samples)**" if _is_jp else "🧪 **Using Sample Data (Mouse, 9 samples)**"
    st.info(msg)

tab_upload, tab_deg, tab_viz, tab_network, tab_meta, tab_export, tab_info = st.tabs([
    "Upload", "DEG", "Visualization", "Network", "🔬 Meta", "Export", "Info"
])

# TAB 1: UPLOAD
with tab_upload:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"

    # ── Mode initialisation ──────────────────────────────────────────────
    if "upload_mode" not in st.session_state:
        st.session_state["upload_mode"] = "single"
    _mode = st.session_state["upload_mode"]

    # ── Mode selection cards ─────────────────────────────────────────────
    _card_col1, _card_col2 = st.columns(2)

    def _card_style(selected: bool) -> str:
        if selected:
            return "border:2px solid #4F6EF7; background:#4F6EF711; border-radius:14px; padding:20px; cursor:pointer;"
        return "border:2px solid #D1D9E6; background:#FAFBFC; border-radius:14px; padding:20px; cursor:pointer;"

    _single_active = _mode == "single"
    _multi_active  = _mode == "multi"

    with _card_col1:
        st.markdown(f"""
<div style='{_card_style(_single_active)}'>
  <div style='font-size:32px; margin-bottom:6px;'>🔬</div>
  <div style='font-weight:800; font-size:17px; color:#1E2A3A; margin-bottom:4px;'>Single Study</div>
  <div style='font-size:12px; font-weight:600; color:#4F6EF7; margin-bottom:10px;'>
    {"1つの実験・複数群の比較" if _is_jp else "One experiment, multiple groups"}
  </div>
  <div style='font-size:13px; color:#4B5563; line-height:1.7; margin-bottom:10px;'>
    {"1つのカウント行列ファイルをアップロードし、Control / Treatment などの群間でDEG解析を行います" if _is_jp else "Upload a single count matrix and run DEG analysis between groups such as Control / Treatment"}
  </div>
  <div style='font-size:11px; color:#6B7280; background:#F3F4F6; border-radius:6px; padding:4px 8px; display:inline-block;'>
    {"例: Control vs Treatment A vs Treatment B" if _is_jp else "e.g. Control vs Treatment A vs Treatment B"}
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("✅ このモードで開始" if _is_jp else "✅ Start with this mode",
                     key="mode_single", use_container_width=True,
                     type="primary" if _single_active else "secondary"):
            st.session_state["upload_mode"] = "single"
            st.rerun()

    with _card_col2:
        st.markdown(f"""
<div style='{_card_style(_multi_active)}'>
  <div style='font-size:32px; margin-bottom:6px;'>📚</div>
  <div style='font-weight:800; font-size:17px; color:#1E2A3A; margin-bottom:4px;'>Multi Study</div>
  <div style='font-size:12px; font-weight:600; color:#4F6EF7; margin-bottom:10px;'>
    {"複数論文・実験の統合メタ解析" if _is_jp else "Meta-analysis across studies"}
  </div>
  <div style='font-size:13px; color:#4B5563; line-height:1.7; margin-bottom:10px;'>
    {"複数のカウント行列ファイルをStudyごとにアップロードし、LFCを統合してVenn図・UpSet図で共通DEGを比較します" if _is_jp else "Upload multiple count matrix files per study and compare common DEGs via LFC integration, Venn, and UpSet plots"}
  </div>
  <div style='font-size:11px; color:#6B7280; background:#F3F4F6; border-radius:6px; padding:4px 8px; display:inline-block;'>
    {"例: アトピー論文 + 乾癬論文 + AEW論文" if _is_jp else "e.g. Atopic + Psoriasis + AEW models"}
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("✅ このモードで開始" if _is_jp else "✅ Start with this mode",
                     key="mode_multi", use_container_width=True,
                     type="primary" if _multi_active else "secondary"):
            st.session_state["upload_mode"] = "multi"
            st.rerun()

    st.write("")

    # ════════════════════════════════════════════════════════════════════
    #  WELCOME GUIDE (Show when no data is loaded)
    # ════════════════════════════════════════════════════════════════════
    if st.session_state["counts_df"] is None:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4F6EF7 0%, #3D5AFE 100%); padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(79, 110, 247, 0.2);">
            <div style="font-size: 24px; font-weight: 800; margin-bottom: 10px;">👋 {'ようこそ！まずは解析の準備をしましょう' if _is_jp else "Welcome! Let's get started with your analysis"}</div>
            <div style="font-size: 15px; opacity: 0.9; margin-bottom: 25px;">{'専門知識がなくても、以下の3ステップで簡単にRNA-seq解析が進められます。' if _is_jp else 'Follow these 3 simple steps to run your RNA-seq analysis without coding.'}</div>
            <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; flex: 1; min-width: 200px;">
                    <div style="font-size: 20px; margin-bottom: 5px;">📁 STEP 1</div>
                    <div style="font-weight: 700;">{'カウント行列をアップロード' if _is_jp else 'Upload Count Matrix'}</div>
                    <div style="font-size: 13px; opacity: 0.8;">{"Excel(CSV)やTSVファイルを選択してLoadを押すだけです。" if _is_jp else 'Select your CSV/TSV file and click Load.'}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; flex: 1; min-width: 200px;">
                    <div style="font-size: 20px; margin-bottom: 5px;">📋 STEP 2</div>
                    <div style="font-weight: 700;">{"サンプルの「群」を設定" if _is_jp else 'Assign Groups'}</div>
                    <div style="font-size: 13px; opacity: 0.8;">{'各サンプルが Control か Treatment かをプルダウンで選びます。' if _is_jp else 'Select group labels for each sample.'}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 12px; flex: 1; min-width: 200px;">
                    <div style="font-size: 20px; margin-bottom: 5px;">🧬 STEP 3</div>
                    <div style="font-weight: 700;">{'DEG解析を実行' if _is_jp else 'Run DEG Analysis'}</div>
                    <div style="font-size: 13px; opacity: 0.8;">{'DEGタブに移動し、「Analyze」ボタンを押せば解析開始！' if _is_jp else 'Go to DEG tab and click Analyze to start.'}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def show_validation_card(df, is_jp):
        with st.container(border=True):
            st.markdown("#### ✅ Data Validation Summary" if not is_jp else "#### ✅ データバリデーション結果")
            
            # Integrated Metrics (Genes, Samples, Zero counts)
            m1, m2, m3 = st.columns(3)
            n_rows = df.shape[0]
            n_cols = df.shape[1]
            zero_rate = (df == 0).sum().sum() / df.size if df.size > 0 else 0
            
            m1.metric("遺伝子数 / Genes" if is_jp else "Genes", f"{n_rows:,}")
            m2.metric("サンプル数 / Samples" if is_jp else "Samples", f"{n_cols}")
            m3.metric("ゼロ率 / Zero rate" if is_jp else "Zero rate", f"{zero_rate:.1%}")
            
            st.divider()
            
            c1, c2 = st.columns(2)
            
            # 1. Integer check
            is_int = np.issubdtype(df.values.dtype, np.integer) or (df.values == df.values.astype(int)).all()
            if is_int:
                c1.markdown(f"- ✅ **{'数値形式' if is_jp else 'Data Type'}**: {'すべて整数です' if is_jp else 'Integers'}")
            else:
                c1.markdown(f"- ❌ **{'数値形式' if is_jp else 'Data Type'}**: " + ("小数が含まれています。カウントデータか確認してください" if is_jp else "Decimals detected. Please check if this is count data."))
            
            # 2. Duplicate check
            dups = df.index.duplicated().sum()
            if dups == 0:
                c1.markdown(f"- ✅ **{'遺伝子名の重複' if is_jp else 'Duplicates'}**: {'なし' if is_jp else 'None'}")
            else:
                c1.markdown(f"- ⚠️ **{'遺伝子名の重複' if is_jp else 'Duplicates'}**: " + (f"{dups}件の重複遺伝子を自動統合しました" if is_jp else f"Merged {dups} duplicate genes"))
            
            # 3. Sample name check
            invalid_cols = [c for c in df.columns if re.search(r'[^a-zA-Z0-9_.]', str(c)) or ' ' in str(c)]
            if not invalid_cols:
                c2.markdown(f"- ✅ **{'サンプル名' if is_jp else 'Sample Names'}**: {'正常' if is_jp else 'Clean'}")
            else:
                c2.markdown(f"- ⚠️ **{'サンプル名' if is_jp else 'Sample Names'}**: " + ("スペースや記号が含まれています。アンダースコアへの置換を推奨します" if is_jp else "Contains spaces or symbols. Underscores recommended."))

    # ════════════════════════════════════════════════════════════════════
    #  SINGLE STUDY MODE
    # ════════════════════════════════════════════════════════════════════
    if _mode == "single":

        # Normalization
        _norm_label = "正規化方法 / Normalization method"
        _norm_opts = ["log1p", "CPM", "TPM", "VST"]
        _norm_sel = st.radio(_norm_label, _norm_opts, horizontal=True, key="norm_method")
        if _is_jp:
            _norm_desc = {
                "log1p": "log(counts + 1) — デフォルト・汎用的",
                "CPM": "Counts Per Million — ライブラリサイズ補正",
                "TPM": "Transcripts Per Million — 遺伝子長情報なしのため CPM と同等の結果になります（参考用）", # BUG FIX: バグ⑦
                "VST": "Variance Stabilizing Transformation — DESeq2ベース・分散安定化",
            }
        else:
            _norm_desc = {
                "log1p": "log(counts + 1) — default, general purpose",
                "CPM": "Counts Per Million — library size correction",
                "TPM": "Transcripts Per Million — fallback to CPM (gene length unavailable; for reference only)", # BUG FIX: バグ⑦
                "VST": "Variance Stabilizing Transformation — DESeq2-based, stabilizes variance",
            }
        st.caption(_norm_desc[_norm_sel])
        if _norm_sel == "TPM": # BUG FIX: バグ⑦
            st.caption("Note: TPM calculation fallbacks to CPM due to missing gene length info.")

        ul, ur = st.columns([1, 1], gap="large")
        with ul:
            btn_text = "🧪 サンプルデータで試す" if _is_jp else "🧪 Try with Sample Data"
            if st.button(btn_text, use_container_width=True):
                with st.status("Generating Sample Data...") as status:
                    cdf, meta = generate_sample_data()
                    st.session_state["counts_df"] = cdf
                    st.session_state["qc_filtered_df"] = cdf
                    st.session_state["metadata"] = meta
                    st.session_state["conditions"] = ["Control", "Treatment_A", "Treatment_B"]
                    st.session_state["sp"] = SPECIES_MAP["Mouse (mmu)"]
                    st.session_state["is_sample_data"] = True
                    st.session_state["deg_results"] = None
                    # Show validation for sample data too
                    st.session_state["last_validation_df"] = cdf
                    status.update(label="✅ Sample Data Loaded! (500 genes, 3 groups)", state="complete", expanded=False)
                st.rerun()

            st.divider()
            uploaded_counts = st.file_uploader(t("count_matrix", lang), type=["csv", "tsv", "txt"])
            if uploaded_counts:
                sp_sel = st.selectbox(t("species", lang), list(SPECIES_MAP.keys()))
                st.session_state["sp"] = SPECIES_MAP[sp_sel]
                id_mode_sel = st.radio(t("gene_id_mode", lang), [t("gene_symbol_opt", lang), t("gene_ids_opt", lang)])
                if st.button("Load", type="primary"):
                    with st.status("🛠️ Processing...", expanded=True) as status:
                        name = uploaded_counts.name.lower()
                        sep = "\t" if name.endswith((".tsv", ".txt")) else ","
                        raw_df = pd.read_csv(uploaded_counts, sep=sep, index_col=0)
                        if id_mode_sel == t("gene_ids_opt", lang):
                            ids = [re.sub(r'\.\d+$', '', str(idx)) for idx in raw_df.index]
                            m = run_online_mapping(ids, "mouse" if st.session_state["sp"]["org"]=="mmu" else "human")
                            raw_df.index = [m.get(i, i) for i in ids]
                        counts_df = raw_df.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
                        if counts_df.index.duplicated().any(): counts_df = counts_df.groupby(counts_df.index).sum()
                        st.session_state["counts_df"] = counts_df
                        st.session_state["metadata"] = None
                        st.session_state["is_sample_data"] = False
                        st.session_state["last_validation_df"] = counts_df
                        status.update(label="✅ Ready!", state="complete", expanded=False)
                        st.rerun()

        with ur:
            if st.session_state.get("last_validation_df") is not None:
                show_validation_card(st.session_state["last_validation_df"], _is_jp)
                if st.button("Close Validation" if not _is_jp else "バリデーションを閉じる"):
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
                st.dataframe(df.head(10), use_container_width=True)
                ng = st.number_input(t("n_groups", lang), 2, 6, 2)
                gnames = [st.text_input(f"Group {i+1}", f"G{i+1}", key=f"gn_{i}") for i in range(ng)]
                sample_map = {s: st.selectbox(f"Assign {s}", gnames, key=f"gs_{s}") for s in df.columns}
                st.session_state["metadata"] = pd.DataFrame.from_dict(sample_map, orient="index", columns=["condition"])
                st.session_state["conditions"] = list(dict.fromkeys(gnames))

                # Low-count filtering
                counts_to_use = df
                genes_removed = 0
                if st.session_state.get("filter_enable", False):
                    thresh = st.session_state["filter_min_count"]
                    m_samples = st.session_state["filter_min_samples"]
                    mask = (df >= thresh).sum(axis=1) >= m_samples
                    counts_to_use = df.loc[mask]
                    genes_removed = df.shape[0] - counts_to_use.shape[0]
                    if st.session_state.get("last_filter_params") != (thresh, m_samples, counts_to_use.shape[0]):
                        st.session_state["last_filter_params"] = (thresh, m_samples, counts_to_use.shape[0])
                        log_analysis("Low-count filtering applied", f"Threshold: {thresh}, Min samples: {m_samples}, Genes removed: {genes_removed}")

                st.session_state["qc_filtered_df"] = counts_to_use

                if st.session_state.get("filter_enable", False):
                    st.info(f"Filtering: {df.shape[0]:,} → {counts_to_use.shape[0]:,} genes ({genes_removed:,} removed)")

                # QC Dashboard
                st.divider()
                st.subheader("📊 QC Dashboard")
                qc_df = st.session_state["qc_filtered_df"]
                _cur_norm = st.session_state.get("norm_method", "log1p")
                if _cur_norm not in ["log1p", "VST"]:
                    if _is_jp:
                        st.info(f"💡 PCA・相関ヒートマップには **log1p** または **VST** が推奨です。現在: **{_cur_norm}**")
                    else:
                        st.info(f"💡 **log1p** or **VST** is recommended for PCA and correlation heatmap. Current: **{_cur_norm}**")

                qct1, qct2 = st.tabs(["Library Size & Detected Genes", "Correlation & PCA"])
                with qct1:
                    lib_size = qc_df.sum(axis=0).reset_index()
                    lib_size.columns = ["Sample", "Total Reads"]
                    log_scale = st.checkbox("Log scale (Library Size)", value=False)
                    fig_lib = px.bar(lib_size, x="Sample", y="Total Reads", title="Library Size per Sample",
                                     template=plotly_template, color="Sample")
                    if log_scale:
                        fig_lib.update_layout(yaxis_type="log")
                    fig_lib.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_lib, use_container_width=True)
                    det_genes = (qc_df > 0).sum(axis=0).reset_index()
                    det_genes.columns = ["Sample", "Detected Genes"]
                    fig_det = px.bar(det_genes, x="Sample", y="Detected Genes", title="Detected Genes per Sample",
                                     template=plotly_template, color="Sample")
                    fig_det.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_det, use_container_width=True)

                with qct2:
                    corr_mat = qc_df.corr()
                    fig_corr = px.imshow(corr_mat, text_auto=".2f", aspect="auto",
                                         color_continuous_scale="Viridis", title="Sample Correlation Heatmap",
                                         template=plotly_template)
                    fig_corr.update_layout(font=dict(family=sel_font, size=fig_font_sz))
                    st.plotly_chart(fig_corr, use_container_width=True)
                    log_qc = np.log1p(qc_df)
                    from sklearn.decomposition import PCA
                    pca_qc = PCA(n_components=2)
                    coords_qc = pca_qc.fit_transform(log_qc.T)
                    pca_df_qc = pd.DataFrame(coords_qc, columns=["PC1", "PC2"], index=qc_df.columns)
                    pca_df_qc["condition"] = st.session_state["metadata"]["condition"]
                    fig_pca_qc = px.scatter(pca_df_qc, x="PC1", y="PC2", color="condition",
                                            text=pca_df_qc.index, title="Sample PCA (QC)",
                                            template=plotly_template)
                    fig_pca_qc.update_traces(textposition='top center')
                    exp_var = pca_qc.explained_variance_ratio_ * 100
                    fig_pca_qc.update_layout(
                        xaxis_title=f"PC1 ({exp_var[0]:.1f}%)",
                        yaxis_title=f"PC2 ({exp_var[1]:.1f}%)",
                        font=dict(family=sel_font, size=fig_font_sz)
                    )
                    st.plotly_chart(fig_pca_qc, use_container_width=True)
            else:
                st.info(t("upload_prompt", lang))

    # ════════════════════════════════════════════════════════════════════
    #  MULTI STUDY MODE
    # ════════════════════════════════════════════════════════════════════
    else:

        # ── Multi Study サンプルデータボタン ────────────────────────────
        _ms_btn_text = "🧪 サンプルデータで試す（Atopic / Psoriasis / AEW）" if _is_jp else "🧪 Try Sample Data (Atopic / Psoriasis / AEW)"
        if st.button(_ms_btn_text, use_container_width=True, key="multi_sample_btn"):
            with st.status("Generating Multi-Study Sample Data...", expanded=True) as _msts:
                _msdata = generate_multi_study_sample_data()
                _merged_counts_list = []
                _merged_meta_list   = []
                _loaded_study_names = []
                for _sname2, _sd in _msdata.items():
                    _merged_counts_list.append(_sd["counts"])
                    _merged_meta_list.append(_sd["metadata"])
                    _loaded_study_names.append(_sname2)
                    st.write(f"✅ {_sname2}: {_sd['counts'].shape[1]} samples, {_sd['counts'].shape[0]} genes")
                _mc = pd.concat(_merged_counts_list, axis=1, join="outer").fillna(0).astype(int)
                _mm = pd.concat(_merged_meta_list, axis=0)
                _mconds = list(dict.fromkeys(_mm["condition"].tolist()))
                st.session_state["counts_df"]          = _mc
                st.session_state["qc_filtered_df"]     = _mc
                st.session_state["metadata"]           = _mm
                st.session_state["conditions"]         = _mconds
                st.session_state["multi_study_names"]  = _loaded_study_names
                st.session_state["upload_mode"]        = "multi"
                st.session_state["is_sample_data"]     = True
                st.session_state["deg_results"]        = None
                st.session_state["last_validation_df"] = _mc
                _msts.update(label="✅ 3 Studies Loaded! (Atopic / Psoriasis / AEW)", state="complete", expanded=False)
            st.rerun()

        # ── File upload ──────────────────────────────────────────────────
        _multi_files = st.file_uploader(
            "📁 Count matrix files (one per study)" if not _is_jp else "📁 カウント行列ファイル（Study ごとに1ファイル）",
            type=["csv", "tsv", "txt"],
            accept_multiple_files=True,
            key="multi_files"
        )

        if _multi_files:
            _study_configs = []
            for _i, _f in enumerate(_multi_files):
                _default_name = _f.name.rsplit(".", 1)[0]
                with st.expander(f"📄 {_f.name}", expanded=True):
                    _sname  = st.text_input("Study 名 / Study name" if _is_jp else "Study name",
                                            value=_default_name, key=f"study_name_{_i}")
                    _sp_sel = st.selectbox("種 / Species" if _is_jp else "Species",
                                           list(SPECIES_MAP.keys()), key=f"study_sp_{_i}")
                    _id_mode = st.radio("Gene ID モード / Gene ID mode" if _is_jp else "Gene ID mode",
                                        ["Gene Symbol", "Ensembl ID"],
                                        horizontal=True, key=f"study_idmode_{_i}")
                    _ng = int(st.number_input("群数 / Number of groups" if _is_jp else "Number of groups",
                                              2, 6, 2, key=f"study_ng_{_i}"))
                    _default_gname_map = {0: "Control", 1: "Disease", 2: "Treatment_A", 3: "Treatment_B", 4: "Group_E", 5: "Group_F"}
                    _gnames = [st.text_input(f"Group {_j+1} name", _default_gname_map.get(_j, f"G{_j+1}"),
                                             key=f"study_gn_{_i}_{_j}") for _j in range(_ng)]

                    # Read columns for sample assignment
                    try:
                        _f.seek(0)
                        _fname_lower = _f.name.lower()
                        _sep_guess = "\t" if _fname_lower.endswith((".tsv", ".txt")) else ","
                        _preview = pd.read_csv(_f, sep=_sep_guess, index_col=0, nrows=0)
                        _f.seek(0)
                        _sample_cols = list(_preview.columns)
                    except Exception:
                        _sample_cols = []

                    _sample_assign = {}
                    if _sample_cols:
                        st.markdown("**サンプル → 群割り当て / Sample → group assignment**" if _is_jp
                                    else "**Sample → group assignment**")
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
            if st.button("📥 全Studyを読み込む / Load All Studies",
                         key="multi_load_btn", type="primary"):
                _all_counts = []
                _all_meta   = []
                _all_conds  = []
                _loaded_names = []

                with st.status("📚 Loading studies...", expanded=True) as _sts:
                    for _cfg in _study_configs:
                        _f2 = _cfg["file"]
                        _f2.seek(0)
                        _fname2 = _f2.name.lower()
                        _sep2 = "\t" if _fname2.endswith((".tsv", ".txt")) else ","
                        _raw = pd.read_csv(_f2, sep=_sep2, index_col=0)

                        if _cfg["id_mode"] == "Ensembl ID":
                            st.write(f"🔗 Mapping {_cfg['name']} Ensembl IDs...")
                            _ids2 = [re.sub(r'\.\d+$', '', str(_x)) for _x in _raw.index]
                            _org2 = "mouse" if _cfg["sp"]["org"] == "mmu" else "human"
                            _map2 = run_online_mapping(_ids2, _org2)
                            _raw.index = [_map2.get(_x, _x) for _x in _ids2]

                        _cnt = _raw.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
                        if _cnt.index.duplicated().any():
                            _cnt = _cnt.groupby(_cnt.index).sum()

                        _meta_rows = {}
                        for _s, _g in _cfg["sample_assign"].items():
                            if _s in _cnt.columns:
                                _meta_rows[_s] = {"condition": _g, "batch": _cfg["name"]}

                        _meta_df = pd.DataFrame.from_dict(_meta_rows, orient="index")
                        _all_counts.append(_cnt)
                        _all_meta.append(_meta_df)
                        _all_conds.extend(_cfg["gnames"])
                        _loaded_names.append(_cfg["name"])
                        st.write(f"  ✅ {_cfg['name']}: {_cnt.shape[0]:,} genes × {_cnt.shape[1]} samples")

                    # Merge
                    _merged_counts = pd.concat(_all_counts, axis=1, join="outer").fillna(0).astype(int)
                    _merged_meta   = pd.concat(_all_meta, axis=0)
                    _merged_conds  = list(dict.fromkeys(_all_conds))

                    st.session_state["counts_df"]        = _merged_counts
                    st.session_state["qc_filtered_df"]   = _merged_counts
                    st.session_state["metadata"]         = _merged_meta
                    st.session_state["conditions"]       = _merged_conds
                    st.session_state["multi_study_names"] = _loaded_names
                    st.session_state["upload_mode"]      = "multi"
                    st.session_state["is_sample_data"]   = False
                    st.session_state["last_validation_df"] = _merged_counts

                    _sts.update(label=f"✅ {len(_loaded_names)} studies loaded!", state="complete", expanded=False)

                st.success(f"✅ {len(_loaded_names)}件のStudyを読み込みました / Loaded {len(_loaded_names)} studies")
                st.rerun()

        if st.session_state.get("last_validation_df") is not None:
            show_validation_card(st.session_state["last_validation_df"], _is_jp)
            if st.button("Close Validation" if not _is_jp else "バリデーションを閉じる", key="multi_close_v"):
                del st.session_state["last_validation_df"]
                st.rerun()

        # ── Post-load summary ────────────────────────────────────────────
        if st.session_state.get("counts_df") is not None and st.session_state.get("multi_study_names"):
            _df_m = st.session_state["counts_df"]
            _meta_m = st.session_state["metadata"]
            if st.session_state.get("last_validation_df") is None:
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("遺伝子数 / Genes" if _is_jp else "Genes",        f"{_df_m.shape[0]:,}")
                _mc2.metric("サンプル数 / Samples" if _is_jp else "Samples",  f"{_df_m.shape[1]:,}")
                _mc3.metric("ゼロ率 / Zero rate" if _is_jp else "Zero rate",  f"{(_df_m == 0).sum().sum() / _df_m.size:.1%}")
            st.dataframe(_df_m.head(10), use_container_width=True)
            st.subheader("📦 Study ごとのサンプル数 / Samples per Study" if _is_jp else "📦 Samples per Study")
            st.dataframe(
                _meta_m["batch"].value_counts().rename_axis("Study").reset_index(name="Samples"),
                use_container_width=True
            )

# TAB 2: DEG
with tab_deg:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["counts_df"] is None:
        if _is_jp:
            st.info("💡 **データがありません**\n\nまずは **Upload** タブでカウント行列をアップロードし、メタデータを設定してください。")
        else:
            st.info("💡 **Empty State**\n\nPlease upload a count matrix and set metadata in the **Upload** tab first.")
    else:
        dl, dr = st.columns([1, 1])
        with dl:
            with st.expander("About DEG analysis", expanded=False):
                if _is_jp:
                    st.markdown("""
**DESeq2** は負の二項分布モデルに基づく差次発現解析の標準的な手法です。正規化・分散推定・仮説検定を一貫して行います。

- **log2FoldChange**: Test群 / Reference群の発現比（log2スケール）
- **padj**: Benjamini-Hochberg法による多重検定補正済みp値
- 各グループに最低2サンプル必要です（3サンプル以上を推奨）
- Cook's distance によるアウトライアー除去を自動実施
""")
                else:
                    st.markdown("""
**DESeq2** is the standard method for differential expression analysis based on a negative binomial distribution model. It performs normalization, dispersion estimation, and hypothesis testing in a unified framework.

- **log2FoldChange**: expression ratio of Test / Reference group (log2 scale)
- **padj**: p-value corrected for multiple testing using the Benjamini-Hochberg method
- At least 2 samples per group are required (3 or more recommended)
- Outlier removal via Cook's distance is applied automatically
""")
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
                    "Study を選択 / Select Study" if _is_jp else "Select Study",
                    list(_study_cond_map.keys()),
                    key="deg_study_sel"
                )
                _conds_in_study = _study_cond_map[_study_sel]
                ref = st.selectbox(t("ref_group", lang), _conds_in_study, key="deg_ref_multi")
                test = st.selectbox(t("test_group", lang), [c for c in _conds_in_study if c != ref], key="deg_test_multi")

                if len(_conds_in_study) < 2:
                    st.warning("⚠️ " + ("このStudyには群が1つしかありません。別のStudyを選んでください。" if _is_jp else "This study has only one group. Please select another study."))
            else:
                ref = st.selectbox(t("ref_group", lang), conds)
                test = st.selectbox(t("test_group", lang), [c for c in conds if c != ref])
            
            import multiprocessing
            max_cores = multiprocessing.cpu_count()
            n_cores = st.slider("Number of CPU cores (Parallel Processing)", 1, max_cores, min(4, max_cores), help="マルチコア処理でPyDESeq2の計算を高速化します。")
            
            # サンプル数警告バッジ
            if _min_samples_per_group < 2:
                _err_msg = ""
                for _g, _n in _group_sample_counts.items():
                    if _n < 2: _err_msg += f"「{_g}」群は{_n}サンプルしかありません。" if _is_jp else f"Group '{_g}' has only {_n} samples. "
                st.error("⛔ " + (f"各群に最低2サンプル必要です。現在、{_err_msg}Uploadタブでサンプルの群割り当てを確認してください。" if _is_jp else f"At least 2 samples per group are required. Currently, {_err_msg}Please check your group assignments in the Upload tab."))
            elif _min_samples_per_group == 2:
                st.warning("⚠️ " + ("各群2サンプルです。3サンプル以上を推奨します（Cook's distance補正がスキップされます）。" if _is_jp else "Only 2 samples per group. 3+ recommended (Cook's distance correction will be skipped)."))
            # Multi Studyモードで異なるStudy間の比較警告
            if _upload_mode_ctrl == "multi" and _meta_ctrl is not None and "batch" in _meta_ctrl.columns:
                _ref_batch  = _meta_ctrl.loc[_meta_ctrl["condition"] == ref,  "batch"].unique().tolist() if ref  in _meta_ctrl["condition"].values else []
                _test_batch = _meta_ctrl.loc[_meta_ctrl["condition"] == test, "batch"].unique().tolist() if test in _meta_ctrl["condition"].values else []
                if _ref_batch and _test_batch and set(_ref_batch) != set(_test_batch):
                    st.error("⛔ " + (f"異なるStudy間の直接比較は推奨されません（{_ref_batch[0]} vs {_test_batch[0]}）。同一Study内の群を選択してください。" if _is_jp else f"Cross-study comparison is not recommended ({_ref_batch[0]} vs {_test_batch[0]}). Please compare groups within the same study."))
            _analyze_disabled = _min_samples_per_group < 2
            if st.button("Analyze", type="primary", disabled=_analyze_disabled):
                meta = st.session_state["metadata"]
                counts_per_group = meta[meta["condition"].isin([ref, test])]["condition"].value_counts()
                if counts_per_group.min() < 2:
                    _err_msg = ""
                    for _g, _n in counts_per_group.items():
                        if _n < 2: _err_msg += f"「{_g}」群は{_n}サンプルしかありません。" if _is_jp else f"Group '{_g}' has only {_n} samples. "
                    st.error("⛔ " + (f"各群に最低2サンプル必要です。現在、{_err_msg}Uploadタブでサンプルの群割り当てを確認してください。" if _is_jp else f"At least 2 samples per group are required. Currently, {_err_msg}Please check your group assignments in the Upload tab."))
                else:
                    try:
                        with st.status("Analyzing...", expanded=True) as status:
                            data_to_analyze = st.session_state.get("qc_filtered_df", st.session_state["counts_df"])
                            res = run_deg(data_to_analyze, st.session_state["metadata"], ref, test, n_cpus=n_cores)
                            
                            st.session_state["deg_results"] = res
                            st.session_state["last_contrast"] = f"{test} vs {ref}"
                            status.update(label="Analysis Complete!", state="complete", expanded=False)
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error("⛔ " + (f"解析中にエラーが発生しました。よくある原因: (1)各群が2サンプル未満 (2)カウント値に負の数が含まれる (3)サンプル名に記号が含まれる。詳細: {e}" if _is_jp else f"An error occurred during analysis. Common causes: (1) less than 2 samples per group, (2) negative counts, (3) special characters in sample names. Details: {e}"))
        with dr:
            if st.session_state["deg_results"] is not None:
                res = st.session_state["deg_results"]
                st.success(f"Contrast: {st.session_state['last_contrast']}")
                up = ((res["padj"] < padj_t) & (res["log2FoldChange"] > lfc_t)).sum()
                dn = ((res["padj"] < padj_t) & (res["log2FoldChange"] < -lfc_t)).sum()
                st.metric("Up", up); st.metric("Down", dn)
                st.dataframe(res.head(100), use_container_width=True)
        # 複数コントラスト一括実行
        st.divider()
        _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
        if _is_jp:
            st.subheader("⚡ 複数コントラスト一括実行")
            st.markdown("実行するペアをチェックして「一括実行」ボタンを押してください。")
        else:
            st.subheader("⚡ Batch DEG Analysis")
            st.markdown("Select the contrasts to run and click 'Run All'.")

        _conds_batch = st.session_state.get("conditions", [])
        _all_pairs = [(r, t) for i, r in enumerate(_conds_batch) for t in _conds_batch[i+1:]]
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
                st.info("💡 Multi Study mode: showing only within-study pairs.")
        if len(_all_pairs) >= 2:
            _selected_pairs = []
            _batch_cols = st.columns(min(3, len(_all_pairs)))
            for _pi, (_pr, _pt) in enumerate(_all_pairs):
                with _batch_cols[_pi % 3]:
                    if st.checkbox(f"{_pt} vs {_pr}", value=True, key=f"batch_chk_{_pr}_{_pt}"):
                        _selected_pairs.append((_pr, _pt))

            if st.button("⚡ Run All Selected" if not _is_jp else "⚡ 選択したペアを一括実行", type="primary", key="batch_run_btn"):
                if not _selected_pairs:
                    st.warning("少なくとも1つのペアを選択してください。/ Select at least one pair." if _is_jp else "Please select at least one pair.")
                else:
                    _batch_results = {}
                    _data_batch = st.session_state.get("qc_filtered_df", st.session_state["counts_df"])
                    with st.status(f"Running {len(_selected_pairs)} contrasts...", expanded=True) as _batch_status:
                        for _br, _bt in _selected_pairs:
                            st.write(f"⏳ {_bt} vs {_br}...")
                            try:
                                _res_batch = run_deg(_data_batch, st.session_state["metadata"], _br, _bt, n_cpus=st.session_state.get("n_cores", 1))
                                _batch_results[f"{_bt}_vs_{_br}"] = _res_batch
                                _up = ((_res_batch["padj"] < padj_t) & (_res_batch["log2FoldChange"] > lfc_t)).sum()
                                _dn = ((_res_batch["padj"] < padj_t) & (_res_batch["log2FoldChange"] < -lfc_t)).sum()
                                st.write(f"✅ {_bt} vs {_br}: Up={_up}, Down={_dn}")
                                log_analysis("Batch DEG", f"Contrast: {_bt} vs {_br}, Up: {_up}, Down: {_dn}")
                            except Exception as _be:
                                st.write(f"❌ {_bt} vs {_br}: {_be}")
                        _batch_status.update(label="完了 / Done", state="complete", expanded=False)

                    st.session_state["batch_deg_results"] = _batch_results

            if st.session_state.get("batch_deg_results"):
                st.markdown("---")
                if _is_jp:
                    st.markdown("**一括実行結果のダウンロード**")
                else:
                    st.markdown("**Download Batch Results**")
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
                    _up_title = "📊 UpSet Plot（コントラスト間DEG比較）" if _is_jp else "📊 UpSet Plot (DEG Overlap Across Contrasts)"
                    st.subheader(_up_title)
                    
                    _dir_label = "方向フィルター" if _is_jp else "Direction filter"
                    _dir_opts = (["Up（上昇）", "Down（低下）", "Both（両方）"] if _is_jp
                                 else ["Up", "Down", "Both"])
                    _upset_dir = st.radio(_dir_label, _dir_opts, horizontal=True, index=2, key="upset_dir")
                    
                    _upset_lbl = "プロット" if _is_jp else "Plot"
                    if st.button(_upset_lbl, key="upset_run_btn"):
                        try:
                            from upsetplot import from_memberships, UpSet
                            
                            _upset_sets = {}
                            for _cn, _rd in st.session_state["batch_deg_results"].items():
                                _rd_clean = _rd.dropna(subset=["padj", "log2FoldChange"])
                                if _upset_dir in ["Up", "Up（上昇）"]:
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"] > lfc_t)].index)
                                elif _upset_dir in ["Down", "Down（低下）"]:
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"] < -lfc_t)].index)
                                else:  # Both
                                    _genes = set(_rd_clean[(_rd_clean["padj"] < padj_t) & (_rd_clean["log2FoldChange"].abs() > lfc_t)].index)
                                _upset_sets[_cn] = _genes
                            
                            # BUG FIX: バグ③
                            if not _upset_sets or all(len(v)==0 for v in _upset_sets.values()):
                                st.warning("共通DEGが見つかりませんでした。" if _is_jp else "No DEGs found with current thresholds.")
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
                                st.warning("共通DEGが見つかりませんでした。" if _is_jp else "No DEGs found with current thresholds.")
                        except ImportError:
                            st.error("pip install upsetplot が必要です / Please run: pip install upsetplot")
        else:
            if _is_jp:
                st.info("一括実行には3条件以上が必要です。")
            else:
                st.info("Batch analysis requires 3 or more conditions.")

        # Multi-group set comparison
        if st.session_state.get("conditions") and len(st.session_state["conditions"]) >= 3:
            st.divider()
            st.subheader(t("venn_title", lang))
            others = [c for c in st.session_state["conditions"] if c != ref]
            v_sel = st.multiselect(t("venn_groups_sel", lang), others, default=others[:2])
            plot_type_venn = st.radio("Plot type", ["Venn", "UpSet"], horizontal=True)

            if st.button(t("venn_run_btn", lang)) and len(v_sel) >= 2:
                try:
                    data_to_analyze = st.session_state.get("qc_filtered_df")
                    if data_to_analyze is None: data_to_analyze = st.session_state["counts_df"]
                    _deg_sets_new = {
                        g: set(run_deg(data_to_analyze, st.session_state["metadata"], ref, g)
                                  .query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index)
                        for g in v_sel
                    }
                    st.session_state["venn_deg_sets"] = _deg_sets_new
                    st.session_state["venn_v_sel"]    = v_sel
                except Exception as e:
                    st.error(f"Venn/UpSet error: {e}")

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
                    st.plotly_chart(_fig_vp, use_container_width=True)
                else:
                    try:
                        from upsetplot import from_memberships, UpSet
                        _memb = [[g for g in _deg_sets if gene in _deg_sets[g]] for gene in _all_union]
                        
                        if not _deg_sets or all(len(v) == 0 for v in _deg_sets.values()):
                            st.warning("現在の閾値ではDEGが見つかりませんでした。" if _is_jp else "No DEGs found with current thresholds.") # A-3
                        else:
                            _udata = from_memberships(_memb)
                            if len(_udata) > 0:
                                _fig_us, _ = plt.subplots(figsize=(12, 6))
                                UpSet(_udata, show_counts=True).plot(fig=_fig_us)
                                st.pyplot(_fig_us)
                                plt.close(_fig_us)
                    except ImportError:
                        st.error("pip install upsetplot")

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
                st.markdown("#### 🧬 " + ("遺伝子一覧 / 領域を選んでKEGG・GO解析が可能" if _is_jp else "Gene List — select a region to run KEGG / GO"))

                _filter_opts = (
                    ["All"] +
                    _venn_groups +
                    [" & ".join(sorted([_a, _b])) for _a, _b in _combs(_venn_groups, 2)] +
                    ([" & ".join(sorted(_venn_groups))] if len(_venn_groups) == 3 else [])
                )
                # ラベルを日本語/英語で
                _filter_labels = {
                    "All": "All（全遺伝子）" if _is_jp else "All genes",
                    " & ".join(sorted(_venn_groups)): "🔴 " + ("3群共通" if _is_jp else "Common to all 3 groups") if len(_venn_groups) == 3 else "",
                }
                _filter_sel = st.selectbox(
                    "領域を選択 / Select region" if _is_jp else "Select region",
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

                st.caption(f"{'選択中' if _is_jp else 'Selected'}: **{_filter_sel}** — {len(_filtered_df)} genes")
                st.dataframe(_filtered_df, use_container_width=True, height=250)

                # CSVダウンロード
                st.download_button(
                    "📥 " + ("遺伝子一覧をCSVでダウンロード" if _is_jp else "Download gene list (CSV)"),
                    _filtered_df.to_csv(index=False),
                    "venn_gene_list.csv", "text/csv",
                    key="dl_venn_genes"
                )

                # ── 選択領域でKEGG / GO ─────────────────────────────────
                st.divider()
                _sel_genes = _filtered_df["Gene"].tolist()
                _n_sel = len(_sel_genes)

                if _n_sel == 0:
                    st.warning("⚠️ " + ("この領域に遺伝子がありません。" if _is_jp else "No genes in this region."))
                else:
                    st.markdown(f"#### 🔬 " + (f"選択領域（{_filter_sel}）の経路解析 — {_n_sel} genes" if _is_jp else f"Pathway Analysis for: {_filter_sel} — {_n_sel} genes"))
                    _venn_enr_col1, _venn_enr_col2 = st.columns(2)

                    # KEGG
                    with _venn_enr_col1:
                        if st.button("🧬 KEGG", type="primary", key="venn_kegg_btn", use_container_width=True):
                            try:
                                import gseapy as gp
                                with st.status("🧬 KEGG..."):
                                    _enr_venn_k = gp.enrichr(
                                        gene_list=_sel_genes,
                                        gene_sets=st.session_state["sp"]["gene_sets_kegg"],
                                        outdir=None
                                    )
                                    st.session_state["venn_enr_kegg"] = _enr_venn_k.results
                            except Exception as _e:
                                st.error(f"KEGG Error: {_e}")

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
                            st.plotly_chart(_fig_vk, use_container_width=True)
                            st.download_button(
                                "📥 KEGG CSV", _vk_df.to_csv(index=False),
                                "venn_kegg.csv", "text/csv", key="dl_venn_kegg"
                            )

                    # GO
                    with _venn_enr_col2:
                        if st.button("🌿 GO", type="primary", key="venn_go_btn", use_container_width=True):
                            try:
                                import gseapy as gp
                                with st.status("🌿 GO..."):
                                    _enr_venn_g = gp.enrichr(
                                        gene_list=_sel_genes,
                                        gene_sets=st.session_state["sp"]["gene_sets_go"],
                                        outdir=None
                                    )
                                    st.session_state["venn_enr_go"] = _enr_venn_g.results
                            except Exception as _e:
                                st.error(f"GO Error: {_e}")

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
                            st.plotly_chart(_fig_vg, use_container_width=True)
                            st.download_button(
                                "📥 GO CSV", _vg_df.to_csv(index=False),
                                "venn_go.csv", "text/csv", key="dl_venn_go"
                            )


# TAB 3: VISUALIZATION
with tab_viz:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["deg_results"] is None:
        if _is_jp:
            st.info("💡 **解析結果がありません**\n\nまずは **DEG** タブで差次発現解析（Analyze）を実行してください。")
        else:
            st.info("💡 **Empty State**\n\nPlease run differential expression analysis in the **DEG** tab first.")
    else:
        lfc_t, padj_t = st.session_state["deg_t"]
        fig_font_sz = st.session_state.get("fig_font_sz", 12)
        v_tab, m_tab, h_tab, gp_tab = st.tabs(
            ["Volcano", "MA Plot", "Heatmap", "Gene Plot"]
        )
        def get_img_bytes(fig, fmt, dpi):
            """Plotly図を画像バイト列に変換。"""
            try:
                return fig.to_image(format=fmt, scale=dpi/72)
            except Exception as e:
                st.error(f"Export error: {e}")
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
            # 遺伝子検索（Volcanoタブ内に移動）
            with st.expander("🔍 " + ("遺伝子をハイライト" if _is_jp else "Highlight a Gene"), expanded=False):
                sc1, sc2 = st.columns([3, 1])
                all_genes = sorted(st.session_state["counts_df"].index.tolist())
                q_gene = sc1.selectbox(t("gene_search_placeholder", lang), [""] + all_genes, key="search_q")
                if sc2.button(t("search_btn", lang), type="primary") and q_gene:
                    st.session_state["viz_highlight"] = q_gene
            with st.expander("About Volcano Plot", expanded=False):
                if _is_jp:
                    st.markdown("""
Volcano Plot は log2FoldChange（x軸）と統計的有意性 −log10(padj)（y軸）を同時に示します。右上が有意な上昇発現、左上が有意な下降発現です。

- 縦・横の破線がそれぞれ padj 閾値・LFC 閾値を示します
- 遺伝子検索で特定の遺伝子を星印でハイライトできます
- 点をホバーすると遺伝子名・統計値を確認できます
""")
                else:
                    st.markdown("""
The Volcano Plot displays log2FoldChange (x-axis) and statistical significance −log10(padj) (y-axis) simultaneously. Genes in the upper-right are significantly up-regulated; upper-left are significantly down-regulated.

- Dashed lines indicate the padj threshold (horizontal) and LFC threshold (vertical)
- Use the gene search box to highlight a specific gene with a star marker
- Hover over any point to view gene name and statistics
""")
            hl = st.session_state.get("search_q", "")
            fig_v = plot_volcano_plotly(st.session_state["deg_results"], padj_t, lfc_t, up_color, down_color, template=plotly_template, font=sel_font, highlight_gene=hl, font_size=fig_font_sz)
            fig_v.update_layout(width=fig_width, height=fig_height, font=dict(family=sel_font, size=fig_font_sz))
            try:
                event = st.plotly_chart(fig_v, use_container_width=True, config=plotly_config, on_select="rerun")
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
                    st.success(f"🎯 なげなわ選択中 ({len(sel_genes)}個): " + ", ".join(sel_genes[:10]) + "...")
                    st.session_state["custom_gene_list"] = sel_genes
            except TypeError:
                st.plotly_chart(fig_v, use_container_width=True, config=plotly_config)
            img_v = get_img_bytes(fig_v, img_format, img_dpi)
            if img_v: st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img_v, f"volcano.{img_format}", key="dl_v_btn")
        
        with m_tab:
            with st.expander("About MA Plot", expanded=False):
                if _is_jp:
                    st.markdown("""
MA Plot は平均発現量（baseMean、x軸・log スケール）と log2FoldChange（y軸）の関係を示します。

- 低発現遺伝子ほどLFCがばらつく傾向があり、フィルタリングの参考になります
- y=0 の破線からの乖離が大きいほど発現変動が大きい遺伝子です
- Volcano Plot と併用することで解析の信頼性を確認できます
""")
                else:
                    st.markdown("""
The MA Plot shows mean expression (baseMean, x-axis, log scale) vs. log2FoldChange (y-axis).

- Low-expression genes tend to show higher LFC variance — useful for filtering decisions
- Greater deviation from the y=0 dashed line indicates larger expression change
- Use alongside the Volcano Plot to verify analysis reliability
""")
            hl = st.session_state.get("search_q", "")
            fig_m = plot_ma_plotly(st.session_state["deg_results"], padj_t, up_color, down_color, template=plotly_template, font=sel_font, highlight_gene=hl, font_size=fig_font_sz)
            fig_m.update_layout(width=fig_width, height=fig_height, font=dict(family=sel_font, size=fig_font_sz))
            st.plotly_chart(fig_m, use_container_width=True, config=plotly_config)
            img_m = get_img_bytes(fig_m, img_format, img_dpi)
            if img_m: st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img_m, f"ma_plot.{img_format}", key="dl_m_btn")



        # — NEW: Heatmap tab
        with h_tab:
            with st.expander("About DEG Heatmap", expanded=False):
                if _is_jp:
                    st.markdown("""
有意DEGの上位N遺伝子について、全サンプルの正規化発現量（現在の設定に連動）を表示します。推奨: **log1p** または **VST**。

- 表示遺伝子数はスライダーで調整できます（デフォルト: 上位50遺伝子）
- padj でソートされた上位遺伝子が選択されます
- 論文の Figure として直接使用できる形式で出力可能です
""")
                else:
                    st.markdown("""
Displays normalized expression of the top N significant DEGs across all samples (linked to current normalization setting). Recommended: **log1p** or **VST**.

- Number of genes shown is adjustable via slider (default: top 50 genes)
- Genes are selected by sorting on padj (most significant first)
- Output can be used directly as a publication-ready figure
""")
            res_h = st.session_state["deg_results"]
            norm_cnt = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"))
            top_n_h = st.slider("Top N genes", 10, 200, 50, key="heatmap_n")
            
            # --- 追加: 表示モードの選択 ---
            hm_mode = st.radio("Display Mode", ["Per Sample", "Group Average"], horizontal=True, key="hm_display_mode")
            
            # Select top N by padj
            top_genes = res_h.dropna(subset=["padj"]).sort_values("padj").head(top_n_h).index
            top_genes = [g for g in top_genes if g in norm_cnt.index]
            if top_genes:
                hm_df = norm_cnt.loc[top_genes]
                
                # --- 追加: 群平均計算ロジック ---
                if hm_mode == "Group Average":
                    meta_h = st.session_state["metadata"]
                    hm_df = hm_df.T.groupby(meta_h["condition"]).mean().T

                # Z-score normalise per gene
                hm_z = hm_df.subtract(hm_df.mean(axis=1), axis=0).divide(hm_df.std(axis=1).replace(0, 1), axis=0)
                fig_h = px.imshow(
                    hm_z,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=0,
                    template=plotly_template,
                    title=f"Top {top_n_h} DEGs — Z-scored log-normalised counts",
                    labels={"x": "Sample", "y": "Gene", "color": "Z-score"},
                )
                fig_h.update_layout(
                    font=dict(family=sel_font, size=fig_font_sz),
                    width=fig_width, height=max(fig_height, top_n_h * 14)
                )
                fig_h.update_yaxes(tickfont_size=max(6, 130 // top_n_h))
                st.plotly_chart(fig_h, use_container_width=True, config=plotly_config)
                img_h = get_img_bytes(fig_h, img_format, img_dpi)
                if img_h:
                    st.download_button(f"\U0001F4E5 Download heatmap ({img_format.upper()})",
                                       img_h, f"heatmap.{img_format}", key="dl_h_btn")
            else:
                st.info("ℹ️ " + ("有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。" if _is_jp else "No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun."))

        # — NEW: Gene Plot tab
        with gp_tab:
            with st.expander("About Gene Plot", expanded=False):
                if _is_jp:
                    st.markdown("""複数の遺伝子を選択してパネル表示（Facet）が可能です。最大12遺伝子まで推奨。

**推奨正規化方法:** log1p または CPM
- **log1p**: 外れ値の影響を抑えた汎用的な可視化に適しています
- **CPM**: ライブラリサイズの違いが大きいサンプル間比較に適しています
""")
                else:
                    st.markdown("""Select multiple genes for panel (facet) display. Up to 12 genes recommended.

**Recommended normalization:** log1p or CPM
- **log1p**: suitable for general visualization, reduces the effect of outliers
- **CPM**: suitable for comparing samples with large differences in library size
""")

            norm_gp = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"))
            meta_gp = st.session_state["metadata"]
            all_g_list = sorted(norm_gp.index.tolist())
            
            gp_genes = st.multiselect("Select genes", all_g_list, 
                                      default=[st.session_state.get("search_q")] if st.session_state.get("search_q") in all_g_list else [all_g_list[0]] if all_g_list else [],
                                      max_selections=12, key="gp_genes_multisel")
            
            gp_type = st.radio("Plot type", ["Boxplot", "Violin"], horizontal=True, key="gp_type_radio")
            
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
                melted = expr_data.melt(id_vars=["Sample", "condition"], var_name="Gene", value_name="log1p(counts)")
                
                cond_palette = st.session_state.get("custom_cond_colors", {})
                
                # Dynamic height based on number of genes (rows)
                n_rows = (len(gp_genes) - 1) // 3 + 1
                dynamic_height = max(fig_height, 300 * n_rows)

                if gp_type == "Boxplot":
                    fig_gp = px.box(melted, x="condition", y="log1p(counts)", color="condition",
                                    facet_col="Gene", facet_col_wrap=3, facet_row_spacing=0.1,
                                    points="all", hover_data=["Sample"],
                                    color_discrete_map=cond_palette, template=plotly_template,
                                    title="Multi-Gene Panel Plot")
                else:
                    fig_gp = px.violin(melted, x="condition", y="log1p(counts)", color="condition",
                                       facet_col="Gene", facet_col_wrap=3,
                                       box=True, points="all", hover_data=["Sample"],
                                       color_discrete_map=cond_palette, template=plotly_template,
                                       title="Multi-Gene Panel Plot")

                fig_gp.update_layout(font=dict(family=sel_font, size=fig_font_sz), 
                                     width=fig_width, height=dynamic_height, showlegend=True)
                fig_gp.for_each_annotation(lambda a: a.update(text=f"<b>{a.text.split('=')[-1]}</b>")) # 遺伝子名を太字に
                
                st.plotly_chart(fig_gp, use_container_width=True, config=plotly_config)
                img_gp = get_img_bytes(fig_gp, img_format, img_dpi)
                if img_gp:
                    file_name_gp = f"gene_plot_panel.{img_format}"
                    st.download_button(f"📥 Download ({img_format.upper()})",
                                       img_gp, file_name_gp, key="dl_gp_btn")

# TAB 4: NETWORK
with tab_network:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    if st.session_state["deg_results"] is None:
        if _is_jp:
            st.info("💡 **解析結果がありません**\n\nまずは **DEG** タブで差次発現解析（Analyze）を実行してください。")
        else:
            st.info("💡 **Empty State**\n\nPlease run differential expression analysis in the **DEG** tab first.")
    else:
        fig_font_sz = st.session_state.get("fig_font_sz", 12)
        st.header("Network & Functional Analysis")
        nt1_k, nt1_g, nt2, nt_string, nt3, nt4, nt_corr = st.tabs(["KEGG", "GO", "GSEA", "STRING", "TF Activity", "Deconvolution", "Gene Correlation"])
        
        with nt1_k:
            st.subheader("KEGG Pathway Enrichment")
            with st.expander("About KEGG enrichment", expanded=False):
                if _is_jp:
                    st.markdown("""
**KEGG (Kyoto Encyclopedia of Genes and Genomes)** は代謝経路・シグナル伝達経路を収録した代表的なデータベースです。  
選択した有意DEGリストをKEGG経路と照合し、**Enrichr (GSEApy)** を用いて過剰表現解析を行います。

- **Combined Score** = ln(*p*-value) × z-score（大きいほど有意）
- **Adjusted P-value** で補正済み
- Up/Down/All の方向で遺伝子セットを絞り込めます
""")
                else:
                    st.markdown("""
**KEGG (Kyoto Encyclopedia of Genes and Genomes)** is a major database of metabolic and signaling pathways. Significant DEGs are tested for over-representation against KEGG pathway gene sets using **Enrichr (GSEApy)**.

- **Combined Score** = ln(p-value) × z-score (higher = more significant)
- **Adjusted P-value** is multiple-testing corrected
- Gene sets can be filtered by direction: Up / Down / All
""")
            direction_k = st.selectbox("Gene set", ["All", "UP Only", "DOWN Only", "Lasso Selected (Custom)"], key="k_dir")
            if _n_sig_degs == 0:
                st.info("ℹ️ " + ("有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。" if _is_jp else "No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun."))
            _kegg_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if st.button("Fetch Pathways", type="primary", disabled=_kegg_disabled):
                try:
                    import gseapy as gp
                    res_deg = st.session_state["deg_results"]
                    if direction_k == "UP Only":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange > {lfc_t}").index.tolist()
                    elif direction_k == "DOWN Only":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange < {-lfc_t}").index.tolist()
                    elif direction_k == "Lasso Selected (Custom)":
                        sig_genes = st.session_state.get("custom_gene_list", [])
                    else:
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()

                    if not sig_genes:
                        st.warning("No significant genes found for the selected direction.")
                    else:
                        with st.status("🧬 Fetching KEGG Results..."):
                            enr = gp.enrichr(gene_list=sig_genes, gene_sets=st.session_state["sp"]["gene_sets_kegg"], outdir=None)
                            st.session_state["enr_kegg"] = enr.results
                        log_analysis("KEGG Run", f"Direction: {direction_k}, Genes: {len(sig_genes)}")
                except Exception as e: st.error(f"KEGG Error: {e}")
            
            if st.session_state["enr_kegg"] is not None:
                st.divider()
                k_c1, k_c2 = st.columns(2)
                with k_c1:
                    top_n_k = st.slider(t("top_n_paths", lang), 5, 50, 10, key="k_top_n")
                with k_c2:
                    pt_k = st.selectbox(t("plot_type", lang), [t("bar_plot", lang), t("dot_plot", lang)], key="k_pt")
                
                df = st.session_state["enr_kegg"].head(top_n_k)
                if pt_k == t("bar_plot", lang):
                    fig = px.bar(df, x='Combined Score', y='Term', orientation='h', title=f"Top {top_n_k} KEGG ({direction_k})", 
                                 color='Adjusted P-value', color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"), template=plotly_template)
                    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=sel_font, size=fig_font_sz))
                else:
                    fig = plot_enrich_dot_plotly(df, f"Top {top_n_k} KEGG ({direction_k})", plotly_template, sel_font, fig_font_sz)
                
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"kegg_plot.{img_format}", key="dl_kegg_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(f"### KEGG Data Table (Top {top_n_k})")
                st.dataframe(df, use_container_width=True)

        with nt1_g:
            st.subheader("GO Biological Process")
            with st.expander("About GO enrichment", expanded=False):
                if _is_jp:
                    st.markdown("""
**GO (Gene Ontology) Biological Process** は遺伝子が関与する生物学的プロセスを階層的に分類したオントロジーです。  
**Enrichr (GSEApy)** を用いて過剰表現解析（ORA）を実行します。

- KEGGより粒度が細かく、細胞内シグナルや分子機能レベルの解釈に適しています
- 同様にCombined Score / Adjusted P-valueで評価
- Up/Down/All の方向でフィルタ可能
""")
                else:
                    st.markdown("""
**GO (Gene Ontology) Biological Process** is a hierarchical ontology classifying the biological processes genes are involved in. Over-representation analysis (ORA) is performed using **Enrichr (GSEApy)**.

- Provides finer resolution than KEGG — well-suited for interpreting intracellular signaling and molecular functions
- Results are evaluated using Combined Score and Adjusted P-value
- Gene sets can be filtered by direction: Up / Down / All
""")
            direction_g = st.selectbox("Gene set", ["All", "UP Only", "DOWN Only", "Lasso Selected (Custom)"], key="g_dir")
            if _n_sig_degs == 0:
                st.info("ℹ️ " + ("有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。" if _is_jp else "No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun."))
            _go_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if st.button("Fetch GO Terms", type="primary", disabled=_go_disabled):
                try:
                    import gseapy as gp
                    res_deg = st.session_state["deg_results"]
                    if direction_g == "UP Only":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange > {lfc_t}").index.tolist()
                    elif direction_g == "DOWN Only":
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange < {-lfc_t}").index.tolist()
                    elif direction_g == "Lasso Selected (Custom)":
                        sig_genes = st.session_state.get("custom_gene_list", [])
                    else:
                        sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()

                    if not sig_genes:
                        st.warning("No significant genes found for the selected direction.")
                    else:
                        with st.status("🌿 Fetching GO Results..."):
                            enr = gp.enrichr(gene_list=sig_genes, gene_sets=st.session_state["sp"]["gene_sets_go"], outdir=None)
                            st.session_state["enr_go"] = enr.results
                        log_analysis("GO Run", f"Direction: {direction_g}, Genes: {len(sig_genes)}")
                except Exception as e: st.error(f"GO Error: {e}")
            
            if st.session_state["enr_go"] is not None:
                st.divider()
                g_c1, g_c2 = st.columns(2)
                with g_c1:
                    top_n_g = st.slider(t("top_n_terms", lang), 5, 50, 10, key="g_top_n")
                with g_c2:
                    pt_g = st.selectbox(t("plot_type", lang), [t("bar_plot", lang), t("dot_plot", lang)], key="g_pt")
                
                df = st.session_state["enr_go"].head(top_n_g)
                if pt_g == t("bar_plot", lang):
                    fig = px.bar(df, x='Combined Score', y='Term', orientation='h', title=f"Top {top_n_g} GO ({direction_g})", 
                                 color='Adjusted P-value', color_continuous_scale=st.session_state.get("enr_cmap", "Viridis_r"), template=plotly_template)
                    fig.update_layout(yaxis={'categoryorder':'total ascending', 'title': ''}, font=dict(family=sel_font, size=fig_font_sz))
                else:
                    fig = plot_enrich_dot_plotly(df, f"Top {top_n_g} GO ({direction_g})", plotly_template, sel_font, fig_font_sz)
                
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"go_plot.{img_format}", key="dl_go_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(f"### GO Data Table (Top {top_n_g})")
                st.dataframe(df, use_container_width=True)

        with nt2:
            st.subheader("Preranked GSEA")
            with st.expander("About GSEA", expanded=False):
                if _is_jp:
                    st.markdown("""
**Gene Set Enrichment Analysis (GSEA)** は、全遺伝子を log2FoldChange でランク付けし、遺伝子セット内の遺伝子が上位または下位に集中しているかを評価します。  
ORA（過剰表現解析）とは異なり、閾値で切り捨てることなく全遺伝子を使用するため、微弱だが一貫したシグナルを検出できます。

- **NES (Normalized Enrichment Score)**: 正値＝上位ランク（up-regulated）、負値＝下位ランク（down-regulated）
- **FDR q-value**: ≤ 0.25 が一般的な有意閾値
- 本実装は **GSEApy prerank** を使用（KEGG gene setsを参照）
""")
                else:
                    st.markdown("""
**Gene Set Enrichment Analysis (GSEA)** ranks all genes by log2FoldChange and tests whether genes in a gene set are concentrated at the top or bottom of the ranking. Unlike ORA, no threshold cutoff is applied — all genes are used, enabling detection of weak but consistent signals.

- **NES (Normalized Enrichment Score)**: positive = enriched at top (up-regulated); negative = enriched at bottom (down-regulated)
- **FDR q-value**: ≤ 0.25 is the conventional significance threshold
- This implementation uses **GSEApy prerank** with KEGG gene sets
""")
            _gsea_disabled = _deg_res_ctrl is None
            if _gsea_disabled:
                st.warning("⚠️ " + ("先にDEG解析を実行してください。" if _is_jp else "Please run DEG analysis first."))
            if st.button("Calculate GSEA", type="primary", disabled=_gsea_disabled):
                try:
                    import gseapy as gp
                    res_deg = st.session_state["deg_results"]
                    # ランクデータの作成
                    rank = res_deg[['log2FoldChange']].sort_values('log2FoldChange', ascending=False).reset_index()
                    rank.columns = ['gene_name', 'score'] # A-13
                    
                    with st.status("📈 Calculating GSEA...", expanded=True):
                        # prerank実行（詳細プロットのために結果オブジェクトを丸ごと保存）
                        pre_res = gp.prerank(rnk=rank, gene_sets=st.session_state["sp"]["gene_sets_kegg"], outdir=None)
                        st.session_state["gsea_results"] = pre_res.res2d
                        st.session_state["gsea_object"] = pre_res
                    st.balloons()
                    st.rerun()
                except Exception as e:
                    st.error(f"GSEA Error: {e}")
            
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
                
                st.plotly_chart(fig, use_container_width=True, config=plotly_config)
                img = get_img_bytes(fig, img_format, img_dpi)
                if isinstance(img, bytes):
                    st.download_button(f"📥 {t('dl_plot', lang)} ({img_format.upper()})", img, f"gsea_plot.{img_format}", key="dl_gsea_btn")
                else:
                    st.error(f"⚠️ {t('format', lang)}: {img_format.upper()} Error. Kaleido needs a restart.")
                
                st.write(f"### GSEA Data Table (Top {top_n_gs})")
                st.dataframe(df, use_container_width=True)
                

        with nt_string:
            st.subheader(t("tab_string", lang))
            string_flavor = st.radio("Network Flavor", ["confidence", "evidence", "actions"], horizontal=True, help="confidence: 線の太さで信頼度を表示, evidence: 根拠の種類ごとに色分け")
            with st.expander("About STRING network", expanded=False):
                if _is_jp:
                    st.markdown("""
**STRING** は実験的・計算的に予測されたタンパク質間相互作用（PPI）を収録するデータベースです。  
有意DEGのうち上位最大30遺伝子を用いてインタラクションネットワークを取得・可視化します。

- エッジの太さが相互作用スコア（信頼度）を反映
- ネットワークは **STRING-db API** からリアルタイム取得（インターネット接続必須）
- 遺伝子名が HGNC シンボルであることを確認してください
""")
                else:
                    st.markdown("""
**STRING** is a database of experimentally and computationally predicted protein-protein interactions (PPI). The top 30 significant DEGs are used to fetch and visualize an interaction network.

- Edge thickness reflects interaction confidence score
- The network is fetched in real time from the **STRING-db API** (internet connection required)
- Gene names must be in HGNC symbol format
""")
            _string_disabled = (_deg_res_ctrl is None or _n_sig_degs == 0)
            if _string_disabled:
                st.info("ℹ️ " + ("有意なDEGが見つかりませんでした。左サイドバーの「LFC threshold」を小さく（例: 0.5）、「padj threshold」を大きく（例: 0.1）してから再実行してみてください。" if _is_jp else "No significant DEGs found. Try decreasing 'LFC threshold' (e.g. 0.5) or increasing 'padj threshold' (e.g. 0.1) in the sidebar and rerun."))
            if st.button(t("string_run_btn", lang), disabled=_string_disabled):
                res_deg = st.session_state["deg_results"]
                # Use top DEGs for network
                sig_genes = res_deg.query(f"padj < {padj_t} and log2FoldChange.abs() > {lfc_t}").index.tolist()
                if not sig_genes:
                    st.warning("No significant genes for network construction.")
                else:
                    with st.status("🔗 Fetching STRING Network..."):
                        img_s = get_string_network_img(sig_genes, st.session_state["sp"]["string_id"], flavor=string_flavor)
                    if img_s:
                        st.image(img_s, caption="STRING Interaction Network (Top Genes)")
                        st.download_button("📥 Download STRING Network Image", img_s, "string_network.png", key="dl_string_btn")
                    else:
                        st.error("Could not fetch STRING network. Check internet or species ID.")
            st.warning(t("string_tip", lang))

        with nt3:
            st.subheader("TF Activity Estimation")

            with st.expander("About TF Activity Estimation", expanded=False):
                if _is_jp:
                    st.markdown("""
転写因子（TF）の活性を遺伝子発現データから推定します。TFそのものの発現量ではなく、標的遺伝子群の発現パターンから間接的に活性を算出するため、発現変動が小さいTFも捕えられます。

**CollecTRI** は実験的に検証されたTF-標的遺伝子の相互作用を収録したデータベースです。エビデンスの質が高い一方、収録TF数は少ない傾向があります。

**DoRothEA** はconfidence levelによってフィルタリングできるデータベースです。Aが最も厳格で実験的証拠に基づき、B→C→Dの順に予測ベースの低信頼性インタラクションが含まれます。なお、マウスデータに対するDoRothEAのカバレッジはCollecTRIと比較して限定的です。結果のTF数が少ない場合はConfidence levelをABCまで幅げることを検討してください。

**ULM（univariate linear model）** は各TFを独立に評価し、高速で解釈しやすい結果が得られます。  
**MLM（multivariate linear model）** は全TFを同時にモデル化し、TF間の共線性を考慮したより保守的な推定が可能です。
""")
                else:
                    st.markdown("""
Estimates transcription factor (TF) activity from gene expression data. Rather than using TF expression levels directly, activity is inferred from the expression patterns of target gene sets — enabling detection of TFs with small expression changes.

**CollecTRI** contains experimentally validated TF–target interactions. It offers high-confidence evidence but covers fewer TFs.

**DoRothEA** supports confidence-level filtering. Level A is the most stringent (experimental evidence only); B, C, and D progressively include lower-confidence, prediction-based interactions. Note: DoRothEA coverage for mouse data is limited compared to CollecTRI. If few TFs are returned, consider expanding the confidence level to ABC.

**ULM (univariate linear model)** evaluates each TF independently — fast and easy to interpret.  
**MLM (multivariate linear model)** models all TFs simultaneously, accounting for collinearity and producing more conservative estimates.
""")

            # — Controls
            tf_c1, tf_c2, tf_c3 = st.columns(3)
            with tf_c1:
                method_sel     = st.selectbox("Method", ["ULM", "MLM"], key="tf_method")
            with tf_c2:
                dorothea_level_sel = st.selectbox("DoRothEA confidence", ["A", "AB", "ABC", "ABCD"], key="tf_dorothea_level")
            with tf_c3:
                min_targets    = st.slider("Min. targets per TF", 5, 30, 10, key="tf_min_n")

            _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
            _tf_disabled = _deg_res_ctrl is None
            if _n_total_samples < 6:
                st.warning("⚠️ " + (f"サンプル数が{_n_total_samples}件です。TF Activity推定は6サンプル以上を推奨します（結果が不安定になる場合があります）。" if _is_jp else f"Only {_n_total_samples} samples. TF Activity estimation recommends 6+ samples for reliable results."))
            if _tf_disabled:
                st.warning("⚠️ " + ("先にDEG解析を実行してください。" if _is_jp else "Please run DEG analysis first."))
            if st.button("Estimate TF Activity", type="primary", disabled=_tf_disabled):
                try:
                    import decoupler as dc
                    sp_org   = st.session_state.get("sp", {}).get("org", "hsa")
                    organism = "human" if sp_org == "hsa" else "mouse"
                    norm_mat = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p")).T  # samples x genes

                    with st.status("🧬 TF Activity Pipeline", expanded=True) as status:
                        # ── decoupler v2.x API ──────────────────────────────
                        st.write("⏳ Step 1/3: Fetching CollecTRI network...")
                        net_collectri = dc.op.collectri(organism=organism, remove_complexes=False)
                        
                        st.write("⏳ Step 2/3: Fetching DoRothEA network...")
                        levels        = list(dorothea_level_sel)  # e.g. ['A','B']
                        net_dorothea  = dc.op.dorothea(organism=organism, levels=levels)

                        st.write(f"⏳ Step 3/3: Calculating activities using {method_sel} (this may take a few minutes)...")
                        if method_sel == "ULM":
                            acts_c, _ = dc.mt.ulm(data=norm_mat, net=net_collectri, tmin=min_targets)
                            acts_d, _ = dc.mt.ulm(data=norm_mat, net=net_dorothea,  tmin=min_targets)
                        else:
                            acts_c, _ = dc.mt.mlm(data=norm_mat, net=net_collectri, tmin=min_targets)
                            acts_d, _ = dc.mt.mlm(data=norm_mat, net=net_dorothea,  tmin=min_targets)
                        
                        status.update(label="✅ TF Activity Estimation Complete!", state="complete", expanded=False)

                    st.session_state["tf_collectri"] = acts_c
                    st.session_state["tf_dorothea"]  = acts_d

                    if acts_d.shape[1] < 5:
                        st.warning(
                            "DoRothEA returned fewer than 5 TFs for this dataset. "
                            "This is expected for mouse data — consider expanding the confidence level (e.g. ABC) "
                            "or using CollecTRI results as the primary reference."
                        )
                    log_analysis("TF Activity", f"Method: {method_sel}, DoRothEA level: {dorothea_level_sel}, organism: {organism}")
                    st.rerun()
                except ImportError:
                    st.error("`decoupler` is not installed. Run: `pip install decoupler`")
                except Exception as e:
                    st.error(f"TF Activity error: {e}")

            # — Results
            if st.session_state["tf_collectri"] is not None:
                acts_c = st.session_state["tf_collectri"]
                acts_d = st.session_state["tf_dorothea"]
                res_c_tab, res_d_tab, res_cons_tab = st.tabs(["CollecTRI", "DoRothEA", "Consensus"])

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
                    st.caption(f"{acts_c.shape[1]} TFs estimated (CollecTRI)")
                    fig_c_tf = _tf_heatmap(acts_c, "TF Activity — CollecTRI (top 20 by mean |activity|)")
                    st.plotly_chart(fig_c_tf, use_container_width=True, config=plotly_config)
                    img_c_tf = get_img_bytes(fig_c_tf, img_format, img_dpi)
                    if img_c_tf:
                        st.download_button("Download plot", img_c_tf, f"tf_collectri.{img_format}", key="dl_ctr_btn")
                    st.download_button("Download CSV (CollecTRI)",
                                       acts_c.to_csv(), "tf_collectri.csv", "text/csv", key="dl_ctr_csv")

                with res_d_tab:
                    st.caption(f"{acts_d.shape[1]} TFs estimated (DoRothEA)")
                    if acts_d.shape[1] < 5:
                        st.warning("Too few TFs to display a meaningful heatmap. Try expanding the DoRothEA confidence level.")
                    else:
                        fig_d_tf = _tf_heatmap(acts_d, "TF Activity — DoRothEA (top 20 by mean |activity|)")
                        st.plotly_chart(fig_d_tf, use_container_width=True, config=plotly_config)
                        img_d_tf = get_img_bytes(fig_d_tf, img_format, img_dpi)
                        if img_d_tf:
                            st.download_button("Download plot", img_d_tf, f"tf_dorothea.{img_format}", key="dl_dor_btn")
                    st.download_button("Download CSV (DoRothEA)",
                                       acts_d.to_csv(), "tf_dorothea.csv", "text/csv", key="dl_dor_csv")

                with res_cons_tab:
                    shared_tfs = list(set(acts_c.columns) & set(acts_d.columns))
                    if len(shared_tfs) < 3:
                        st.warning(
                            "Not enough shared TFs between CollecTRI and DoRothEA to generate a consensus plot. "
                            "This is common with mouse data and strict confidence levels."
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
                        st.plotly_chart(fig_scatter, use_container_width=True, config=plotly_config)

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
                        st.plotly_chart(fig_cons_hm, use_container_width=True, config=plotly_config)
                        img_cons = get_img_bytes(fig_cons_hm, img_format, img_dpi)
                        if img_cons:
                            st.download_button("Download consensus heatmap", img_cons,
                                               f"tf_consensus.{img_format}", key="dl_cons_btn")

        with nt4:
            st.subheader("Immune Deconvolution")
            with st.expander("About Immune Deconvolution", expanded=False):
                if _is_jp:
                    st.markdown("""
バルクRNA-seqデータから各サンプルの免疫細胞組成を推定します。Nu-SVR（Support Vector Regression）を用いてリファレンスの細胞型シグネチャに最も近い組成を最適化します（CIBERSORTに準じた実装）。

- **ICE**: 実験的に検証されたマーカー遺伝子を使用したオープンリファレンス
- **Human Cell Atlas**: HCA由来の細胞型シグネチャ（ヒトデータに最適）
- **LM22 / カスタム**: 独自リファレンスをCSVで持ち込み可能（行=遺伝子、列=細胞型）
- 遺伝子シンボルがHGNC形式であることを確認してください
- マウスデータの場合、ビルトインリファレンスはヒト由来のため重複遺伝子数が少なくなる可能性があります（本ツールで自動変換を試みます）
""")
                else:
                    st.markdown("""
Estimates immune cell composition of each sample from bulk RNA-seq data. Uses Nu-SVR (Support Vector Regression) to find the cell type mixture that best fits the reference signatures (CIBERSORT-like implementation).

- **ICE**: open reference using experimentally validated marker genes
- **Human Cell Atlas**: cell type signatures derived from HCA (best suited for human data)
- **LM22 / Custom**: upload your own reference as a CSV (rows = genes, columns = cell types)
- Gene symbols must be in HGNC format
- For mouse data, built-in references are human-derived. This tool will automatically attempt symbol conversion (e.g. CD8A -> Cd8a).
""")

            # — Controls
            ref_choice = st.radio(
                "Reference matrix",
                ["Built-in: ICE (Immune Cell Expression)",
                 "Built-in: Human Cell Atlas signatures",
                 "Built-in: Neural & Glial cells (Brain/Spinal Cord)",
                 "Upload LM22 or custom CSV"],
                key="decon_ref_choice"
            )
            normalize = st.checkbox("Normalize fractions to sum to 1 per sample", value=True, key="decon_norm")

            ref_df_upload = None
            if ref_choice == "Upload LM22 or custom CSV":
                up_ref = st.file_uploader("Reference CSV (rows=genes, columns=cell types)",
                                          type=["csv"], key="decon_ref_upload")
                if up_ref is not None:
                    ref_df_upload = pd.read_csv(up_ref, index_col=0)

            _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
            _decon_disabled = st.session_state.get("counts_df") is None
            if _n_total_samples < 4:
                st.warning("⚠️ " + (f"サンプル数が{_n_total_samples}件です。デコンボリューションは4サンプル以上を推奨します。" if _is_jp else f"Only {_n_total_samples} samples. Deconvolution recommends 4+ samples."))
            if st.button("Deconvolve", type="primary", disabled=_decon_disabled):
                try:
                    from sklearn.svm import NuSVR
                    from sklearn.preprocessing import StandardScaler

                    # Build or load reference DataFrame
                    ref_dict = None
                    is_builtin = False
                    if ref_choice == "Built-in: ICE (Immune Cell Expression)":
                        ref_dict = ICE_REFERENCE
                        is_builtin = True
                    elif ref_choice == "Built-in: Human Cell Atlas signatures":
                        ref_dict = HCA_REFERENCE
                        is_builtin = True
                    elif ref_choice == "Built-in: Neural & Glial cells (Brain/Spinal Cord)":
                        ref_dict = NEURAL_REFERENCE
                        is_builtin = True
                    else:
                        if ref_df_upload is None:
                            st.error("Please upload a reference CSV first.")
                            st.stop()
                        ref_df_upload.index = ref_df_upload.index.str.strip()
                        ref_df_used = ref_df_upload.astype(float)

                    if ref_dict is not None:
                        all_markers = sorted(set(g for genes in ref_dict.values() for g in genes))
                        ref_df_used = pd.DataFrame(
                            {ct: [1.0 if g in genes else 0.0 for g in all_markers]
                             for ct, genes in ref_dict.items()},
                            index=all_markers
                        )
                    
                    # --- マウス対応: ビルトインリファレンスの遺伝子名を変換 ---
                    _sp_org = st.session_state.get("sp", {}).get("org", "human")
                    if is_builtin and _sp_org == "mmu":
                        # 例: CD8A -> Cd8a
                        ref_df_used.index = [str(g).capitalize() for g in ref_df_used.index]
                        # 重複が生じた場合は最大値（存在フラグ）をとる
                        ref_df_used = ref_df_used.groupby(level=0).max()
                    
                    # 堅牢な共通遺伝子抽出
                    counts_decon = st.session_state["counts_df"].copy()
                    counts_decon.index = counts_decon.index.str.strip()
                    ref_df_used.index = ref_df_used.index.str.strip()
                    
                    common_genes = counts_decon.index.intersection(ref_df_used.index)
                    
                    if len(common_genes) < 5:
                        st.error("⛔ " + (f"リファレンスと一致する遺伝子が {len(common_genes)} 件しかありません。マウスデータなら「Cd8a」、ヒトデータなら「CD8A」形式の遺伝子名になっているか確認してください。" if _is_jp else f"Only {len(common_genes)} genes match the reference. Ensure gene names are in 'CD8A' format for human or 'Cd8a' for mouse."))
                    else:
                        X = ref_df_used.loc[common_genes].values.astype(float)
                        results = {}
                        with st.status(f"Running Nu-SVR deconvolution ({len(counts_decon.columns)} samples on {len(common_genes)} markers)..."):
                            for sample in counts_decon.columns:
                                y = np.log1p(counts_decon.loc[common_genes, sample].values.astype(float))
                                scaler_X = StandardScaler()
                                scaler_y = StandardScaler()
                                X_s = scaler_X.fit_transform(X)
                                y_s = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
                                svr = NuSVR(kernel="linear", nu=0.5)
                                svr.fit(X_s, y_s)
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
                    st.error(f"Deconvolution error: {e}")

            # — Results
            if st.session_state["ciber_results"] is not None:
                ciber_df = st.session_state["ciber_results"]
                st.caption(f"Estimated fractions for {ciber_df.shape[0]} samples, {ciber_df.shape[1]} cell types")

                # Attach condition info for coloring
                meta_dc = st.session_state.get("metadata")
                if meta_dc is not None:
                    ciber_plot = ciber_df.copy()
                    ciber_plot.index.name = "Sample"
                    ciber_plot = ciber_plot.reset_index().melt(id_vars="Sample",
                                                                var_name="Cell type",
                                                                value_name="Fraction")
                    fig_decon = px.bar(
                        ciber_plot, x="Sample", y="Fraction", color="Cell type",
                        barmode="stack",
                        template=plotly_template,
                        title="Immune cell composition (Nu-SVR deconvolution)"
                    )
                    fig_decon.update_layout(font=dict(family=sel_font, size=fig_font_sz),
                                            width=fig_width, height=fig_height,
                                            xaxis_tickangle=-40)
                    st.plotly_chart(fig_decon, use_container_width=True, config=plotly_config)
                    img_dc = get_img_bytes(fig_decon, img_format, img_dpi)
                    if img_dc:
                        st.download_button("Download plot", img_dc,
                                           f"deconvolution.{img_format}", key="dl_dc_btn")

                st.download_button("Download CSV (fractions)",
                                   ciber_df.to_csv(), "deconvolution_fractions.csv",
                                   "text/csv", key="dl_dc_csv")
                st.dataframe(ciber_df.style.format("{:.3f}"), use_container_width=True)

        with nt_corr:
            _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
            if _is_jp:
                st.subheader("🔗 遺伝子相関解析")
            else:
                st.subheader("🔗 Gene Correlation Analysis")
            with st.expander("About Gene Correlation Analysis", expanded=False):
                if _is_jp:
                    st.markdown("""
2つの遺伝子の発現量の相関を散布図で可視化します。Pearson・Spearman相関係数とp値を表示します。

**推奨正規化方法:** log1p または VST
- **log1p**: 汎用的・手軽に使えます
- **VST**: 分散が安定しているため、より信頼性の高い相関解析が可能です（論文品質）
""")
                else:
                    st.markdown("""
Visualize the correlation between two genes as a scatter plot. Displays Pearson and Spearman correlation coefficients with p-values.

**Recommended normalization:** log1p or VST
- **log1p**: general purpose, easy to use
- **VST**: more reliable correlation analysis due to stabilized variance (publication quality)
""")

            _norm_corr = normalize_counts(st.session_state["counts_df"], st.session_state.get("norm_method", "log1p"))
            _all_genes_corr = sorted(_norm_corr.index.tolist())
            _cc1, _cc2 = st.columns(2)
            _gene_a = _cc1.selectbox("Gene A", _all_genes_corr, key="corr_gene_a")
            _gene_b = _cc2.selectbox("Gene B", _all_genes_corr, index=min(1, len(_all_genes_corr)-1), key="corr_gene_b")
            _corr_method = st.radio("Correlation method", ["Pearson", "Spearman"], horizontal=True, key="corr_method")

            _n_total_samples = st.session_state["counts_df"].shape[1] if st.session_state.get("counts_df") is not None else 0
            if _n_total_samples < 4:
                st.warning("⚠️ " + (f"サンプル数が{_n_total_samples}件です。相関解析は4サンプル以上を推奨します（相関係数の信頼性が低下します）。" if _is_jp else f"Only {_n_total_samples} samples. Correlation analysis recommends 4+ samples for reliable results."))
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

                if _corr_method == "Pearson":
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
                st.plotly_chart(_fig_corr, use_container_width=True, config=plotly_config)

                _rc1, _rc2, _rc3 = st.columns(3)
                _rc1.metric(f"{_corr_method} r", f"{_r:.4f}")
                _rc2.metric("p-value", f"{_p:.2e}")
                _rc3.metric("Significant", "Yes ✅" if _p < 0.05 else "No ❌")

                _img_corr = get_img_bytes(_fig_corr, img_format, img_dpi)
                if _img_corr:
                    st.download_button(f"📥 Download ({img_format.upper()})", _img_corr, f"correlation_{_gene_a}_{_gene_b}.{img_format}", key="dl_corr_btn")
            else:
                st.info("Gene A と Gene B に異なる遺伝子を選択してください。/ Please select two different genes." if _is_jp else "Please select two different genes for Gene A and Gene B.")

# TAB 5: META-ANALYSIS
with tab_meta:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    _batch = st.session_state.get("batch_deg_results", {})
    
    _n_studies_meta   = len(st.session_state.get("multi_study_names", []))
    _n_contrasts_meta = len(st.session_state.get("batch_deg_results", {}))

    if len(_batch) < 2:
        st.info(
            "💡 DEGタブで2つ以上のコントラストを一括実行してください" if _is_jp else
            "💡 Please run 2 or more contrasts in the DEG tab first"
        )
    else:
        _contrast_names = list(_batch.keys())
        _n_contrasts = len(_contrast_names)
        padj_t  = st.session_state.get("padj_t",  0.05)
        lfc_t   = st.session_state.get("lfc_t",   1.0)

        # ---- 2-2. LFC/padj integrated matrices ----
        _lfc_frames  = []
        _padj_frames = []
        for _cn, _rd in _batch.items():
            _lfc_frames.append(_rd[["log2FoldChange"]].rename(columns={"log2FoldChange": _cn}))
            _padj_frames.append(_rd[["padj"]].rename(columns={"padj": _cn}))
        _lfc_mat  = pd.concat(_lfc_frames,  axis=1, join="outer")
        _padj_mat = pd.concat(_padj_frames, axis=1, join="outer")

        # ---- 2-3. Filtering UI ----
        st.header("🔬 Meta-Analysis" if not _is_jp else "🔬 メタ解析（複数コントラスト統合）")
        _mc1, _mc2, _mc3 = st.columns(3)
        with _mc1:
            _dir_opts = ["上昇（Up）", "低下（Down）", "両方（Both）"] if _is_jp else ["Up", "Down", "Both"]
            _meta_dir = st.radio(
                "方向フィルター" if _is_jp else "Direction",
                _dir_opts, index=2, horizontal=True, key="meta_dir"
            )
        with _mc2:
            _meta_min_n = st.slider(
                "有意コントラスト数（最少）" if _is_jp else "Min contrasts (significant)",
                1, _n_contrasts, min(2, _n_contrasts), key="meta_min_n"
            )
        with _mc3:
            st.write("")
            _meta_run = st.button("プロット" if _is_jp else "Plot", key="meta_plot_btn", type="primary")

        # Download full LFC matrix (unfiltered)
        st.download_button(
            "📥 LFC統合マトリクス CSV" if _is_jp else "📥 Download LFC Matrix CSV",
            _lfc_mat.to_csv(),
            "lfc_integrated_matrix.csv",
            key="meta_dl_lfc"
        )

        if _meta_run:
            # ---- 2-4. Filter ----
            _sig_bool = pd.DataFrame(index=_padj_mat.index)
            for _cn in _contrast_names:
                _p  = _padj_mat[_cn].fillna(1.0)
                _lf = _lfc_mat[_cn].fillna(0.0)
                if _meta_dir in ["Up", "上昇（Up）"]:
                    _sig_bool[_cn] = (_p < padj_t) & (_lf > lfc_t)
                elif _meta_dir in ["Down", "低下（Down）"]:
                    _sig_bool[_cn] = (_p < padj_t) & (_lf < -lfc_t)
                else:
                    _sig_bool[_cn] = (_p < padj_t) & (_lf.abs() > lfc_t)

            _n_sig = _sig_bool.sum(axis=1)
            _keep  = _n_sig[_n_sig >= _meta_min_n].index

            if len(_keep) == 0:
                st.warning(
                    "指定された条件で有意な遙伝子が見つかりませんでした。" if _is_jp else
                    "No genes met the specified criteria."
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
                    title=("🔥 LFC統合ヒートマップ" if _is_jp else "🔥 LFC Integrated Heatmap"),
                    height=_heat_h,
                    yaxis={"tickfont": {"size": 9}},
                    xaxis_title="Contrast",
                    yaxis_title="Gene",
                    font=dict(family=sel_font, size=fig_font_sz),
                    template=plotly_template,
                )
                st.plotly_chart(_fig_heat, use_container_width=True)

                # ---- 2-6. Venn / UpSet (自動切替) ----
                st.divider()
                _deg_sets = {
                    _cn: set(_sig_bool[_sig_bool[_cn]].index)
                    for _cn in _contrast_names
                }
                _dir_label = str(_meta_dir)

                if _n_contrasts <= 3:
                    _plot_mode_meta = "Venn"
                    st.caption("💡 " + ("コントラスト数が3以下のためVenn図を表示します。" if _is_jp else "≤3 contrasts → showing Venn diagram."))
                else:
                    _plot_mode_meta = "UpSet"
                    st.caption("💡 " + ("コントラスト数が4以上のためUpSet図を表示します。" if _is_jp else "≥4 contrasts → showing UpSet plot."))

                if not _deg_sets or all(len(v) == 0 for v in _deg_sets.values()):
                    st.warning("現在の閾値ではDEGが見つかりませんでした。" if _is_jp else "No DEGs found with current thresholds.") # A-11
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
        st.subheader("📦 Package Export")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for f, d in collect_all_results().items():
                z.writestr(f, d)
        st.download_button(
            "📦 Download Results ZIP",
            buf.getvalue(),
            "results.zip",
            "application/zip",
            key="dl_zip_btn"
        ) # A-16
        
        st.divider()
        st.subheader("📝 Reproducibility")
        import json
        repro_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "species": st.session_state.get("sp", {}).get("org", "unknown"), # A-12
            "parameters": {
                "lfc_threshold": st.session_state.get("lfc_t", 1.0),
                "padj_threshold": st.session_state.get("padj_t", 0.05),
                "font": st.session_state.get("selected_font", "Arial"),
                "filtering": {
                    "enabled": st.session_state.get("filter_enable", False),
                    "threshold": st.session_state.get("filter_min_count", 10),
                    "min_samples": st.session_state.get("filter_min_samples", 2)
                }
            },
            "log": st.session_state.get("analysis_log", [])
        }
        st.download_button(t("dl_report", lang), json.dumps(repro_data, indent=2), "reproducibility_report.json", "application/json")

        st.divider()
        st.subheader("📄 Publication-Ready Report")
        if _is_jp:
            st.markdown("論文の Methods セクションにそのままコピペできる英語テキストと、解析パラメータをまとめたHTMLレポートを出力します。")
        else:
            st.markdown("Generates an HTML report containing analysis parameters and a publication-ready Methods section in English.")
        
        # HTML Report Generation
        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>Bulk RNA-seq Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; color: #333; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                ul {{ list-style-type: square; }}
                .methods {{ background-color: #f8f9fa; padding: 20px; border-left: 5px solid #007bff; margin: 20px 0; font-family: "Times New Roman", Times, serif; font-size: 11pt; }}
                .ref {{ font-size: 10pt; color: #555; }}
            </style>
        </head>
        <body>
            <h1>Bulk RNA-seq Analysis Report</h1>
            <p><strong>Generated on:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Software Version:</strong> {APP_VERSION}</p>
            
            <h2>1. Analysis Parameters</h2>
            <ul>
                <li><strong>Log2 Fold Change Threshold:</strong> {st.session_state.get('lfc_t', 1.0)}</li>
                <li><strong>Adjusted P-value Threshold:</strong> {st.session_state.get('padj_t', 0.05)}</li>
            </ul>

            <h2>2. DEG Summary</h2>
            <p><strong>Contrast:</strong> {st.session_state.get('last_contrast', 'N/A')}</p>
            <ul>
                <li><strong>Up-regulated genes:</strong> {((st.session_state["deg_results"]["padj"].fillna(1.0) < st.session_state["padj_t"]) & (st.session_state["deg_results"]["log2FoldChange"].fillna(0.0) > st.session_state["lfc_t"])).sum()}</li>
                <li><strong>Down-regulated genes:</strong> {((st.session_state["deg_results"]["padj"].fillna(1.0) < st.session_state["padj_t"]) & (st.session_state["deg_results"]["log2FoldChange"].fillna(0.0) < -st.session_state["lfc_t"])).sum()}</li>
            </ul>

            <h2>3. Draft Methods Section (Ready to copy)</h2>
            <p>The text below can be used as a draft for your manuscript's Methods section.</p>
            <div class="methods">
                <p><strong>RNA-seq data processing and differential expression analysis:</strong><br>
                Count matrices were analyzed using the Bulk RNA-seq Analyzer application (v{APP_VERSION}). Differential expression analysis was performed using the PyDESeq2 package [1,2]. Genes with an adjusted p-value &lt; {st.session_state.get('padj_t', 0.05)} and |log2FoldChange| &gt; {st.session_state.get('lfc_t', 1.0)} were considered statistically significant. 
                Pathway enrichment analysis and Gene Set Enrichment Analysis (GSEA) were conducted using GSEApy [3-5]. Transcription factor activity was estimated using decoupleR [6] with the CollecTRI and DoRothEA databases [7,8]. Immune cell deconvolution was performed using a Support Vector Regression (Nu-SVR) approach [9].</p>
                
                <p class="ref"><strong>References:</strong><br>
                [1] Muzellec, L. et al., Bioinformatics (2023).<br>
                [2] Love, M. I. et al., Genome Biology (2014).<br>
                [3] Fang, Z. et al., Bioinformatics (2022).<br>
                [4] Kuleshov, M. V. et al., Nucleic Acids Research (2016).<br>
                [5] Subramanian, A. et al., PNAS (2005).<br>
                [6] Badia-i-Mompel, A. et al., Bioinformatics Advances (2022).<br>
                [7] Müller-Dott, S. et al., Nucleic Acids Research (2023).<br>
                [8] Garcia-Alonso, L. et al., Genome Research (2019).<br>
                [9] Newman, A. M. et al., Nature Methods (2015).</p>
            </div>
        </body>
        </html>
        """
        st.download_button("📥 Download HTML Report", html_content, "Analysis_Report.html", "text/html", key="dl_html_report")
    else:
        _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
        if _is_jp:
            st.info("💡 **出力するデータがありません**\n\nまずは **DEG** タブで解析を実行して結果を生成してください。")
        else:
            st.info("💡 **No data to export**\n\nPlease run analysis in the **DEG** tab to generate results first.")

# TAB 6: INFO
with tab_info:
    _is_jp = st.session_state.get("lang_display", "日本語") == "日本語"
    st.header(t("tab_info", lang))
    st.divider()
    st.subheader("Environment & Versions")
    
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

    # ─── Anaconda Setup Guide ────────────────────────────────────
    st.divider()
    st.subheader("🐍 Anaconda でローカル実行する方法 / How to Run Locally with Anaconda")

    if _is_jp:
        st.markdown("""
このアプリの解析をご自身の PC で再現するための手順です。  
**Anaconda (Python 3.10 以上)** がインストールされていることを前提としています。
""")
    else:
        st.markdown("""
Follow these steps to reproduce the analyses on your own PC.  
Requires **Anaconda (Python 3.10 or later)** to be installed.
""")

    with st.expander("① 仮想環境の作成 / Create a virtual environment", expanded=False):
        st.code("""# 仮想環境を作成（Python 3.10 推奨）
conda create -n bulkrnaseq python=3.10 -y
conda activate bulkrnaseq""", language="bash")

    with st.expander("② パッケージのインストール / Install packages", expanded=False):
        if _is_jp:
            st.markdown("以下のコマンドを **順番通り** に実行してください。")
        else:
            st.markdown("Run the following commands **in order**.")
        st.code("""# 基本パッケージ
pip install streamlit pandas numpy matplotlib seaborn plotly requests

# 差次発現解析
pip install pydeseq2

# 富化解析・GSEA
pip install gseapy

# 機械学習（PCA・免疫細胞デコンボリューション）
pip install scikit-learn

# 図の書き出し（PDF/SVG/PNG）
pip install kaleido

# 転写因子活性解析
pip install decoupler omnipath""", language="bash")
        st.info("💡 `decoupler` と `omnipath` は転写因子（TF）解析タブを使用する場合のみ必要です。")

    with st.expander("③ アプリファイルの配置 / Place app files", expanded=False):
        if _is_jp:
            st.markdown("""
以下のファイルを同じフォルダに置いてください。

| ファイル | 説明 |
|---|---|
| `Bulk_RNAseq_Analyzer.py` | メインアプリ |
| `i18n.py` | 多言語対応モジュール |

フォルダ構成例:
""")
        else:
            st.markdown("""
Place the following files in the same folder.

| File | Description |
|---|---|
| `Bulk_RNAseq_Analyzer.py` | Main app |
| `i18n.py` | Internationalization module |

Example folder structure:
""")
        st.code("""my_bulkrnaseq_app/
├── Bulk_RNAseq_Analyzer.py
└── i18n.py""", language="text")

    with st.expander("④ アプリの起動 / Launch the app", expanded=False):
        if _is_jp:
            st.markdown("フォルダに移動してから以下を実行してください。")
        else:
            st.markdown("Navigate to the folder and run the following.")
        st.code("""cd my_bulkrnaseq_app
conda activate bulkrnaseq
streamlit run Bulk_RNAseq_Analyzer.py""", language="bash")
        if _is_jp:
            st.success("ブラウザが自動で開き、アプリが表示されます（通常 http://localhost:8501）。")
        else:
            st.success("Your browser will open automatically (usually at http://localhost:8501).")

    with st.expander("⑤ 入力ファイルの形式 / Input file format", expanded=False):
        if _is_jp:
            st.markdown("""
**カウント行列（counts matrix）**  
- 行：遺伝子名（Gene Symbol または Ensembl ID）  
- 列：サンプル名  
- ファイル形式：`.csv` または `.tsv`

例:
""")
        else:
            st.markdown("""
**Count matrix**  
- Rows: gene names (Gene Symbol or Ensembl ID)  
- Columns: sample names  
- File format: `.csv` or `.tsv`

Example:
""")
        st.code("""gene_id,Sample1,Sample2,Sample3,Sample4
Gapdh,1523,1489,1601,1472
Actb,3201,3089,3310,3185
Tnf,45,12,98,88
Il6,23,8,201,156""", language="text")
        if _is_jp:
            st.markdown("""
**メタデータ（metadata）**  
- 行：サンプル名（カウント行列の列名と一致）  
- 列：`condition`（必須）  
- ファイル形式：`.csv`

例:
""")
        else:
            st.markdown("""
**Metadata**  
- Rows: sample names (must match column names in count matrix)  
- Required column: `condition`  
- File format: `.csv`

Example:
""")
        st.code("""sample,condition
Sample1,Control
Sample2,Control
Sample3,Treatment
Sample4,Treatment""", language="text")

    with st.expander("⑥ トラブルシューティング / Troubleshooting", expanded=False):
        if _is_jp:
            st.markdown("""
| エラー | 対処法 |
|---|---|
| `ModuleNotFoundError: No module named 'i18n'` | `i18n.py` が同じフォルダにあるか確認してください |
| `ModuleNotFoundError: No module named 'pydeseq2'` | `pip install pydeseq2` を再実行してください |
| `kaleido` で図がエクスポートできない | `pip install kaleido==0.2.1` で旧バージョンを試してください |
| `decoupler` インストールエラー | `pip install decoupler omnipath --no-deps` を試してください |
| Streamlit が古くてエラー | `pip install --upgrade streamlit` を実行してください |
""")
        else:
            st.markdown("""
| Error | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'i18n'` | Make sure `i18n.py` is in the same folder |
| `ModuleNotFoundError: No module named 'pydeseq2'` | Re-run `pip install pydeseq2` |
| Cannot export figures with `kaleido` | Try `pip install kaleido==0.2.1` |
| `decoupler` install error | Try `pip install decoupler omnipath --no-deps` |
| Streamlit version errors | Run `pip install --upgrade streamlit` |
""")

    with st.expander("📋 requirements.txt (コピーして使用 / Copy and use)", expanded=False):
        if _is_jp:
            st.markdown("`requirements.txt` としてアプリと同じフォルダに保存し、`pip install -r requirements.txt` でまとめてインストールできます。")
        else:
            st.markdown("Save as `requirements.txt` in the same folder, then run `pip install -r requirements.txt` to install all at once.")
        st.code("""streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.18.0
requests>=2.28.0
pydeseq2>=0.4.0
gseapy>=1.1.0
scikit-learn>=1.3.0
kaleido==0.2.1
decoupler>=1.6.0
omnipath>=1.0.0""", language="text")

    st.divider()
    st.info(t("notebook_desc", lang))
    if not st.session_state["analysis_log"]:
        st.write("履歴がありません。解析を実行してください。" if _is_jp else "No analysis history yet. Please run an analysis.") # BUG FIX: バグ⑧
    else:
        for entry in st.session_state["analysis_log"]:
            with st.expander(f"[{entry['time']}] {entry['action']}", expanded=True):
                st.markdown(entry["details"].replace("\n", "  \n"))
        
        all_res = collect_all_results()
        if "Analysis_Notebook.md" in all_res:
            st.download_button(t("dl_notebook", lang), all_res["Analysis_Notebook.md"], "Analysis_Notebook.md")

    # ─── References ─────────────────────────────────────────────
    st.divider()
    st.subheader("📚 References / 引用文献")
    if _is_jp:
        st.write("本アプリの解析は以下のツールとデータベースに支えられています。論文発表の際は、使用した機能に応じて該当する文献を引用してください。")
    else:
        st.write("This application relies on the following tools and databases. Please cite the respective publications based on the features you used.")
    with st.expander("Show all references / 引用文献一覧を表示", expanded=False):
        st.markdown("""
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
""")

    # ─── Contact Form ───────────────────────────────────────────
    st.divider()
    st.header("Contact / お問い合わせ")
    if _is_jp:
        st.write("バグ報告・機能リクエスト・ご質問はこちらからお気軽にどうぞ。")
    else:
        st.write("Feel free to reach out for bug reports, feature requests, or any questions.")
    st.caption("Developer: Motoki Morita | ORCID: [0009-0008-9402-9387](https://orcid.org/0009-0008-9402-9387)")

    cf_category = st.selectbox(
        "Category / カテゴリ",
        ["Bug report", "Feature request", "Question", "Other"],
        key="cf_category"
    )
    cf_version  = st.text_input("App version", value=APP_VERSION, disabled=True, key="cf_version")
    cf_message  = st.text_area("Message / メッセージ", height=150, key="cf_message")

    if st.button("Open in mail app / メールアプリで送信", type="primary", key="cf_send_btn"):
        if not cf_message:
            st.warning("Please enter a message. / メッセージを入力してください。")
        else:
            import urllib.parse
            subject = f"[BulkSeq Analyzer v{APP_VERSION}] {cf_category}"
            body = (
                f"Category: {cf_category}\n"
                f"App version: {APP_VERSION}\n"
                f"\n"
                f"Message:\n{cf_message}"
            )
            mailto = (
                f"mailto:kudp19101@gmail.com"
                f"?subject={urllib.parse.quote(subject)}"
                f"&body={urllib.parse.quote(body)}"
            )
            st.markdown(
                f'<a href="{mailto}" target="_blank">'
                f'<button style="background:#1F6FEB;color:white;border:none;padding:8px 18px;'
                f'border-radius:6px;cursor:pointer;font-size:14px;">'
                f'Click here to open mail app / メールアプリを開く</button></a>',
                unsafe_allow_html=True
            )
            st.caption("ボタンを押すとメールアプリが開きます。送信内容を確認してから送信してください。 / Your mail app will open with the message pre-filled. Please review before sending.")
