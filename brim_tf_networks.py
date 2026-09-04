"""Load the transcription-factor networks bundled with BRIM."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NETWORK_DIR = Path(__file__).resolve().parent / "references" / "tf_networks"
SUPPORTED_ORGANISMS = {"human", "mouse"}


def _validate_organism(organism: str) -> str:
    normalized = str(organism).strip().lower()
    if normalized not in SUPPORTED_ORGANISMS:
        raise ValueError(
            f"Unsupported organism {organism!r}; expected 'human' or 'mouse'."
        )
    return normalized


@lru_cache(maxsize=4)
def _read_network(resource: str, organism: str) -> pd.DataFrame:
    organism = _validate_organism(organism)
    path = NETWORK_DIR / f"{resource}_{organism}.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Bundled TF network is missing: {path}. Reinstall BRIM or restore "
            "the references/tf_networks folder."
        )

    network = pd.read_csv(path)
    required = {"source", "target", "weight"}
    missing = required.difference(network.columns)
    if missing:
        raise ValueError(
            f"Bundled TF network {path.name} is invalid; missing columns: "
            f"{', '.join(sorted(missing))}."
        )
    if network.empty or network[["source", "target", "weight"]].isna().any().any():
        raise ValueError(f"Bundled TF network {path.name} is empty or contains missing values.")
    if network.duplicated(["source", "target"]).any():
        raise ValueError(f"Bundled TF network {path.name} contains duplicate TF-target pairs.")
    if not np.isfinite(pd.to_numeric(network["weight"], errors="coerce")).all():
        raise ValueError(f"Bundled TF network {path.name} contains invalid weights.")
    return network


def load_collectri_network(organism: str) -> pd.DataFrame:
    """Return a private copy of the bundled CollecTRI network."""
    return _read_network("collectri", _validate_organism(organism)).copy()


def load_dorothea_network(
    organism: str, levels: Iterable[str] = ("A", "B", "C", "D")
) -> pd.DataFrame:
    """Return a private copy of DoRothEA filtered to confidence levels."""
    selected = tuple(dict.fromkeys(str(level).upper() for level in levels))
    invalid = set(selected).difference({"A", "B", "C", "D"})
    if not selected or invalid:
        raise ValueError("DoRothEA levels must contain one or more of A, B, C, and D.")

    network = _read_network("dorothea", _validate_organism(organism))
    if "confidence" not in network.columns:
        raise ValueError("Bundled DoRothEA network is missing the confidence column.")
    return network.loc[network["confidence"].isin(selected)].copy()


def infer_tf_activity(
    data: pd.DataFrame,
    network: pd.DataFrame,
    method: str,
    min_targets: int,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Infer TF activity, safely falling back when MLM is not identifiable."""
    import decoupler as dc

    selected = str(method).upper()
    if selected == "ULM":
        scores, pvalues = dc.mt.ulm(data=data, net=network, tmin=min_targets)
        return scores, pvalues, "ULM"
    if selected != "MLM":
        raise ValueError("TF activity method must be 'ULM' or 'MLM'.")

    try:
        scores, pvalues = dc.mt.mlm(data=data, net=network, tmin=min_targets)
        return scores, pvalues, "MLM"
    except (np.linalg.LinAlgError, AssertionError) as error:
        if isinstance(error, AssertionError) and "Could not fit a multivariate linear model" not in str(error):
            raise
        # Some translated networks contain indistinguishable regulator target
        # profiles. A pseudoinverse would report non-identifiable MLM effects;
        # independent ULM fits are safer and reproducible for this case.
        scores, pvalues = dc.mt.ulm(data=data, net=network, tmin=min_targets)
        return scores, pvalues, "ULM fallback"
