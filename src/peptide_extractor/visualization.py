"""Unified visualization framework for peptide_extractor results.

Provides individual plotting functions for PeptideEntry, ExtractionEvaluation,
PaperRelevance, and PaperMetadata records, plus a single unified `visualize()`
interface that validates (model, operation) compatibility, fetches records via
`get_records(**db_kwargs)`, and renders/saves the requested diagrams.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sqlmodel import Session, SQLModel

from peptide_extractor.db.database import _resolve_model, get_records
from peptide_extractor.db.extraction_evaluation import ExtractionEvaluation
from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.db.paper_relevance import PaperRelevance
from peptide_extractor.db.peptide_entry import PeptideEntry

logger = logging.getLogger(__name__)


class VisualizationCompatibilityError(ValueError):
    """Raised when a visualization operation is incompatible with the target data type."""


class VisualizationOperation(str, Enum):
    """Supported visualization operations across pipeline result tables."""

    # PeptideEntry operations
    PEPTIDE_COLUMN_COMPLETION = "peptide_column_completion"
    PEPTIDE_LENGTH_DISTRIBUTION = "peptide_length_distribution"
    EXTRACTIONS_PER_PAPER = "extractions_per_paper"
    PEPTIDE_MORPHOLOGY_DISTRIBUTION = "peptide_morphology_distribution"
    PEPTIDE_RUN_COMPARISON = "peptide_run_comparison"
    PEPTIDEMINER_CATEGORICAL_ACCURACY = "peptideminer_categorical_accuracy"
    PEPTIDEMINER_NUMERICAL_MAE = "peptideminer_numerical_mae"

    # ExtractionEvaluation operations
    EVALUATION_BOXPLOT = "evaluation_boxplot"
    EVALUATION_SUMMARY_RANGE = "evaluation_summary_range"
    EVALUATION_DISTRIBUTION = "evaluation_distribution"

    # PaperRelevance operations
    RELEVANCE_CONFIDENCE_HIST = "relevance_confidence_hist"
    RELEVANCE_OUTCOME_COUNTS = "relevance_outcome_counts"

    # PaperMetadata operations
    PAPER_STATUS_COUNTS = "paper_status_counts"


@dataclass(frozen=True)
class VisualizationSpec:
    """Metadata and handler binding for a visualization operation."""

    operation: VisualizationOperation
    supported_models: Tuple[Type[SQLModel], ...]
    handler: Callable[..., Figure]
    default_filename: str
    description: str


@dataclass
class VisualizationResult:
    """Output container for a single rendered visualization."""

    operation: VisualizationOperation
    model_name: str
    record_count: int
    figure: Figure
    output_path: Optional[Path] = None


# Default PeptideEntry columns inspected for completion rate
DEFAULT_PEPTIDE_COMPLETION_COLUMNS: Tuple[str, ...] = (
    "sequence",
    "length",
    "type",
    "n_terminal",
    "c_terminal",
    "temperature",
    "ph",
    "peptide_concentration",
    "solution",
    "solvent",
    "volume_ratio",
    "incubation_period",
    "peptide_conjugate_mixture",
    "conjugate_partner",
    "linear_cyclic",
    "chirality",
    "detection_method",
    "preparation_protocol",
    "evidence_span",
)

DEFAULT_EVALUATION_METRICS: Tuple[str, ...] = (
    "groundness_rate",
    "faithfulness_rate",
    "completeness_rate",
)


# ---------------------------------------------------------------------------
# Data conversion & validation helpers
# ---------------------------------------------------------------------------


def _serialize_value(val: Any) -> Any:
    if isinstance(val, Enum):
        return val.value
    return val


def records_to_dataframe(records: Sequence[SQLModel]) -> pd.DataFrame:
    """Convert a sequence of SQLModel instances into a cleaned pandas DataFrame."""
    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        raw = rec.model_dump()
        rows.append({k: _serialize_value(v) for k, v in raw.items()})

    df = pd.DataFrame(rows)
    return df.replace(r"^\s*$", np.nan, regex=True)


def resolve_operation(
    operation: Union[VisualizationOperation, str],
) -> VisualizationOperation:
    """Resolve an enum or string name to a VisualizationOperation."""
    if isinstance(operation, VisualizationOperation):
        return operation
    if isinstance(operation, str):
        normalized = operation.strip().lower()
        try:
            return VisualizationOperation(normalized)
        except ValueError as exc:
            available = [op.value for op in VisualizationOperation]
            raise ValueError(
                f"Unknown visualization operation '{operation}'. "
                f"Available operations: {available}"
            ) from exc
    raise TypeError(
        f"Expected VisualizationOperation or str, got {type(operation).__name__}"
    )


def resolve_visualization_model(
    model: Union[Type[SQLModel], str],
) -> Type[SQLModel]:
    """Resolve a SQLModel class or table name string for visualization."""
    return _resolve_model(model)


def get_supported_operations(
    model: Optional[Union[Type[SQLModel], str]] = None,
) -> List[VisualizationOperation]:
    """Return all registered visualization operations, optionally filtered by model."""
    if model is None:
        return list(VISUALIZATION_REGISTRY.keys())
    resolved_model = resolve_visualization_model(model)
    return [
        op
        for op, spec in VISUALIZATION_REGISTRY.items()
        if resolved_model in spec.supported_models
    ]


def validate_operations_for_model(
    operations: Union[
        VisualizationOperation,
        str,
        Sequence[Union[VisualizationOperation, str]],
    ],
    model: Union[Type[SQLModel], str],
) -> List[VisualizationOperation]:
    """Normalize and validate that every requested operation supports `model`."""
    resolved_model = resolve_visualization_model(model)

    if isinstance(operations, (VisualizationOperation, str)):
        op_list = [operations]
    else:
        op_list = list(operations)

    if not op_list:
        raise ValueError("At least one visualization operation must be provided.")

    resolved_ops: List[VisualizationOperation] = []
    for op in op_list:
        resolved_op = resolve_operation(op)
        spec = VISUALIZATION_REGISTRY[resolved_op]
        if resolved_model not in spec.supported_models:
            supported_names = [m.__name__ for m in spec.supported_models]
            valid_for_model = [
                o.value for o in get_supported_operations(resolved_model)
            ]
            raise VisualizationCompatibilityError(
                f"Operation '{resolved_op.value}' is not compatible with data type "
                f"'{resolved_model.__name__}'. Supported data types for this operation: "
                f"{supported_names}. Valid operations for '{resolved_model.__name__}': "
                f"{valid_for_model}"
            )
        resolved_ops.append(resolved_op)

    return resolved_ops


# ---------------------------------------------------------------------------
# Individual plotting functions: PeptideEntry
# ---------------------------------------------------------------------------


def plot_peptide_column_completion(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    group_by: Optional[str] = None,
    figsize: Tuple[float, float] = (9, 6),
    title: Optional[str] = None,
) -> Figure:
    """Plot percentage of non-null/non-empty values across PeptideEntry columns."""
    cols = [c for c in (columns or DEFAULT_PEPTIDE_COMPLETION_COLUMNS) if c in df.columns]
    if not cols:
        raise ValueError("No valid peptide columns found in DataFrame to plot.")

    fig, ax = plt.subplots(figsize=figsize)

    if group_by and group_by in df.columns and df[group_by].nunique(dropna=True) > 1:
        grouped = (
            df.groupby(group_by, dropna=False)[cols]
            .apply(lambda g: g.notna().mean() * 100)
            .reset_index()
        )
        melted = grouped.melt(
            id_vars=[group_by], var_name="column", value_name="non_null_percentage"
        )
        melted[group_by] = melted[group_by].astype(str)
        sns.barplot(
            data=melted,
            x="non_null_percentage",
            y="column",
            hue=group_by,
            ax=ax,
        )
    else:
        completion_pct = (df[cols].notna().mean() * 100).reset_index()
        completion_pct.columns = ["column", "non_null_percentage"]
        sns.barplot(
            data=completion_pct,
            x="non_null_percentage",
            y="column",
            color="steelblue",
            ax=ax,
        )
        for p in ax.patches:
            width = p.get_width()
            ax.annotate(
                f"{width:.1f}%",
                (width, p.get_y() + p.get_height() / 2.0),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )

    ax.set_xlim(0, 105)
    ax.set_xlabel("Completion Rate (%)")
    ax.set_ylabel("Columns")
    ax.set_title(title or "Percentage of Valid (Non-Empty / Non-Null) Data by Column")
    sns.despine(ax=ax, left=True, bottom=False)
    fig.tight_layout()
    return fig


def plot_peptide_length_distribution(
    df: pd.DataFrame,
    min_count: int = 1,
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
) -> Figure:
    """Plot distribution of peptide sequence lengths."""
    if "length" not in df.columns:
        raise ValueError("Column 'length' not found in DataFrame.")

    clean = df.dropna(subset=["length"]).copy()
    counts = clean["length"].value_counts()
    valid_lengths = sorted(counts[counts >= min_count].index.tolist())
    if not valid_lengths:
        raise ValueError(
            f"No peptide lengths have at least {min_count} occurrences."
        )

    df_filtered = clean[clean["length"].isin(valid_lengths)]
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        data=df_filtered,
        x="length",
        order=valid_lengths,
        color="steelblue",
        ax=ax,
    )
    ax.set_xlabel("Peptide Length (residues)")
    ax.set_ylabel("Count")
    ax.set_title(
        title
        or (
            f"Peptide Length Distribution (min_count={min_count})"
            if min_count > 1
            else "Peptide Length Distribution"
        )
    )
    if ax.containers:
        ax.bar_label(ax.containers[0], fontsize=8)
    fig.tight_layout()
    return fig


def plot_extractions_per_paper(
    df: pd.DataFrame,
    id_col: Optional[str] = None,
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot distribution of extracted PeptideEntry counts per paper."""
    col = id_col
    if col is None:
        if "pmid" in df.columns and df["pmid"].notna().any():
            col = "pmid"
        elif "paper_id" in df.columns:
            col = "paper_id"
        else:
            raise ValueError("Neither 'pmid' nor 'paper_id' found in DataFrame.")

    occurrences = df[col].dropna().value_counts()
    if occurrences.empty:
        raise ValueError(f"No non-null values found for '{col}'.")

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        x=occurrences.values,
        color="steelblue",
        edgecolor="black",
        ax=ax,
    )
    ax.set_xlabel("Extraction count per paper")
    ax.set_ylabel("Number of papers")
    ax.set_title(title or f"Extractions per Paper ({col})")
    if ax.containers:
        ax.bar_label(ax.containers[0])
    fig.tight_layout()
    return fig


def plot_peptide_morphology_distribution(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> Figure:
    """Plot counts of nanostructure morphology outcomes (`type`)."""
    if "type" not in df.columns:
        raise ValueError("Column 'type' not found in DataFrame.")

    series = df["type"].fillna("missing").astype(str)
    order = series.value_counts().index.tolist()

    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(
        y=series,
        order=order,
        color="steelblue",
        ax=ax,
    )
    ax.set_xlabel("Count")
    ax.set_ylabel("Morphology Type")
    ax.set_title(title or "Peptide Self-Assembly Morphology Distribution")
    if ax.containers:
        ax.bar_label(ax.containers[0], padding=3)
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_peptide_run_comparison(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    figsize: Tuple[float, float] = (12, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Compare extraction volume, unique papers, and mean completion across runs."""
    if "run_identifier" not in df.columns:
        raise ValueError("Column 'run_identifier' not found in DataFrame.")

    work = df.copy()
    work["run_label"] = work["run_identifier"].fillna("unknown").astype(str)
    cols = [c for c in (columns or DEFAULT_PEPTIDE_COMPLETION_COLUMNS) if c in work.columns]
    paper_col = "pmid" if "pmid" in work.columns and work["pmid"].notna().any() else "paper_id"

    summary_rows = []
    for run_val, grp in work.groupby("run_label", sort=True):
        mean_comp = float(grp[cols].notna().mean().mean() * 100) if cols else 0.0
        unique_papers = int(grp[paper_col].nunique(dropna=True)) if paper_col in grp.columns else 0
        summary_rows.append(
            {
                "run_identifier": run_val,
                "entries": len(grp),
                "unique_papers": unique_papers,
                "mean_completion_pct": mean_comp,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.barplot(data=summary_df, x="run_identifier", y="entries", color="steelblue", ax=axes[0])
    axes[0].set_title("Total Extracted Entries")
    axes[0].set_xlabel("Run Identifier")
    axes[0].set_ylabel("Entries")
    if axes[0].containers:
        axes[0].bar_label(axes[0].containers[0])

    sns.barplot(data=summary_df, x="run_identifier", y="unique_papers", color="teal", ax=axes[1])
    axes[1].set_title("Unique Papers")
    axes[1].set_xlabel("Run Identifier")
    axes[1].set_ylabel("Papers")
    if axes[1].containers:
        axes[1].bar_label(axes[1].containers[0])

    sns.barplot(
        data=summary_df, x="run_identifier", y="mean_completion_pct", color="slateblue", ax=axes[2]
    )
    axes[2].set_ylim(0, 105)
    axes[2].set_title("Mean Field Completion (%)")
    axes[2].set_xlabel("Run Identifier")
    axes[2].set_ylabel("Completion (%)")
    if axes[2].containers:
        axes[2].bar_label(axes[2].containers[0], fmt="%.1f%%")

    for a in axes:
        a.tick_params(axis="x", rotation=20)

    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Individual plotting functions: PeptideMiner Paper Benchmark (Fig. 4c & 4d)
# ---------------------------------------------------------------------------


def _clean_df_val(val: Any) -> Any:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if pd.isna(val):
        return None
    return val


def _df_to_peptide_entries(df: pd.DataFrame) -> List[PeptideEntry]:
    """Reconstruct lightweight PeptideEntry objects from a DataFrame."""
    from peptide_extractor.db.peptide_entry import ExtractionMode

    entries: List[PeptideEntry] = []
    for row in df.to_dict(orient="records"):
        mode_raw = _clean_df_val(row.get("extraction_mode"))
        mode = (
            ExtractionMode(str(mode_raw).lower())
            if mode_raw is not None
            else ExtractionMode.MANUAL
        )
        seq = str(_clean_df_val(row.get("sequence")) or "")
        raw_len = _clean_df_val(row.get("length"))
        raw_run = _clean_df_val(row.get("run_identifier"))
        entries.append(
            PeptideEntry(
                pmid=str(_clean_df_val(row.get("pmid")) or ""),
                sequence=seq,
                length=int(raw_len) if raw_len is not None else len(seq),
                type=_clean_df_val(row.get("type")),
                n_terminal=_clean_df_val(row.get("n_terminal")),
                c_terminal=_clean_df_val(row.get("c_terminal")),
                temperature=_clean_df_val(row.get("temperature")),
                ph=_clean_df_val(row.get("ph")),
                peptide_concentration=_clean_df_val(row.get("peptide_concentration")),
                solution=_clean_df_val(row.get("solution")),
                solvent=_clean_df_val(row.get("solvent")),
                volume_ratio=_clean_df_val(row.get("volume_ratio")),
                incubation_period=_clean_df_val(row.get("incubation_period")),
                peptide_conjugate_mixture=_clean_df_val(
                    row.get("peptide_conjugate_mixture")
                ),
                conjugate_partner=_clean_df_val(row.get("conjugate_partner")),
                linear_cyclic=_clean_df_val(row.get("linear_cyclic")),
                detection_method=_clean_df_val(row.get("detection_method")),
                extraction_mode=mode,
                model=_clean_df_val(row.get("model")),
                run_identifier=int(raw_run) if raw_run is not None else None,
            )
        )
    return entries


def _build_peptideminer_reports_from_df(
    df: pd.DataFrame,
    *,
    include_paper_baselines: bool = True,
    paper_baselines: Optional[Sequence[str]] = None,
    run_labels: Optional[Dict[Any, str]] = None,
) -> Dict[str, Any]:
    """Compute PeptideMinerBenchmarkReport per LLM run (+ optional paper baselines)."""
    from peptide_extractor.db.peptide_entry import ExtractionMode
    from peptide_extractor.golden_benchmark import (
        compare_runs_peptideminer,
        load_peptideminer_paper_baselines,
    )

    entries = _df_to_peptide_entries(df)
    manual = [e for e in entries if e.extraction_mode == ExtractionMode.MANUAL]
    if not manual:
        raise ValueError(
            "No manual golden-dataset PeptideEntry rows (extraction_mode='manual') "
            "found in DataFrame. Include run_identifier=0 in your query."
        )

    reports: Dict[str, Any] = {}
    if include_paper_baselines:
        allowed = set(paper_baselines) if paper_baselines is not None else None
        for base_label, base_entries in load_peptideminer_paper_baselines().items():
            if allowed is not None and base_label not in allowed:
                continue
            rep = compare_runs_peptideminer(manual, base_entries)
            if rep.llm_rows_evaluated > 0:
                reports[base_label] = rep

    llm_by_run: Dict[Any, List[PeptideEntry]] = {}
    for e in entries:
        if e.extraction_mode == ExtractionMode.LLM:
            llm_by_run.setdefault(e.run_identifier, []).append(e)

    labels_map = run_labels or {}
    for run_id, run_entries in llm_by_run.items():
        if run_id in labels_map:
            label = labels_map[run_id]
        else:
            models = sorted({e.model for e in run_entries if e.model})
            if models and run_id is not None:
                label = f"{models[0]} (run {run_id})"
            elif models:
                label = models[0]
            elif run_id is not None:
                label = f"LLM (run {run_id})"
            else:
                label = "LLM"
        reports[label] = compare_runs_peptideminer(manual, run_entries)

    if not reports:
        raise ValueError(
            "No LLM runs or PeptideMiner paper baselines available to plot."
        )
    return reports


def plot_peptideminer_categorical_accuracy(
    df: pd.DataFrame,
    *,
    include_paper_baselines: bool = True,
    paper_baselines: Optional[Sequence[str]] = None,
    use_normalized: bool = True,
    run_labels: Optional[Dict[Any, str]] = None,
    reports: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (12, 5.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot grouped bar chart of categorical feature extraction accuracy (PeptideMiner Fig. 4c).

    Operation: ``VisualizationOperation.PEPTIDEMINER_CATEGORICAL_ACCURACY`` (model: ``PeptideEntry``).
    """
    resolved_reports = (
        reports
        if reports is not None
        else _build_peptideminer_reports_from_df(
            df,
            include_paper_baselines=include_paper_baselines,
            paper_baselines=paper_baselines,
            run_labels=run_labels,
        )
    )

    rows = []
    for model_label, rep in resolved_reports.items():
        avg_acc = (
            rep.average_categorical_normalized_accuracy
            if use_normalized
            else rep.average_categorical_exact_accuracy
        )
        legend_label = f"{model_label} (avg {avg_acc * 100:.1f}%)"
        for field_name, st in rep.categorical_stats.items():
            acc = (st.normalized_accuracy if use_normalized else st.exact_accuracy) * 100.0
            rows.append(
                {
                    "Category": field_name,
                    "Accuracy (%)": acc,
                    "Model": legend_label,
                }
            )

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=plot_df,
        x="Category",
        y="Accuracy (%)",
        hue="Model",
        ax=ax,
    )
    ax.set_ylim(0, 112)
    ax.set_xlabel("Category")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        title
        or (
            "Categorical Feature Extraction Accuracy (PeptideMiner Protocol)"
            if use_normalized
            else "Categorical Feature Extraction Exact Accuracy (PeptideMiner Protocol)"
        ),
        pad=42 if len(resolved_reports) > 3 else 32,
    )
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8, padding=2)
    ax.legend(
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2 if len(resolved_reports) > 3 else min(3, len(resolved_reports)),
        frameon=True,
        fontsize=8,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_peptideminer_numerical_mae(
    df: pd.DataFrame,
    *,
    include_paper_baselines: bool = True,
    paper_baselines: Optional[Sequence[str]] = None,
    normalized: bool = True,
    run_labels: Optional[Dict[Any, str]] = None,
    reports: Optional[Dict[str, Any]] = None,
    figsize: Tuple[float, float] = (11, 5),
    title: Optional[str] = None,
) -> Figure:
    """Plot violin distribution of numerical extraction MAE (PeptideMiner Fig. 4d).

    Operation: ``VisualizationOperation.PEPTIDEMINER_NUMERICAL_MAE`` (model: ``PeptideEntry``).
    """
    resolved_reports = (
        reports
        if reports is not None
        else _build_peptideminer_reports_from_df(
            df,
            include_paper_baselines=include_paper_baselines,
            paper_baselines=paper_baselines,
            run_labels=run_labels,
        )
    )

    rows = []
    for model_label, rep in resolved_reports.items():
        for field_name, st in rep.numerical_stats.items():
            errs = st.normalized_errors if normalized else st.raw_errors
            for err in errs:
                rows.append(
                    {
                        "Category": field_name,
                        "MAE": err,
                        "Model": model_label,
                    }
                )

    if not rows:
        raise ValueError("No non-null numerical errors available to plot.")

    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=plot_df,
        x="Category",
        y="MAE",
        hue="Model",
        cut=0,
        inner="quartile",
        density_norm="width",
        ax=ax,
    )
    ax.set_xlabel("Category")
    ax.set_ylabel("Normalized MAE [0-1]" if normalized else "Raw Absolute Error")
    ax.set_title(
        title
        or (
            "Numerical Feature Extraction Normalized MAE (PeptideMiner Protocol)"
            if normalized
            else "Numerical Feature Extraction Raw Error (PeptideMiner Protocol)"
        ),
        pad=42 if len(resolved_reports) > 3 else 32,
    )
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(
        title=None,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2 if len(resolved_reports) > 3 else min(3, len(resolved_reports)),
        frameon=True,
        fontsize=8,
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Individual plotting functions: ExtractionEvaluation
# ---------------------------------------------------------------------------


def _melt_evaluation_metrics(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    max_rate: Optional[float] = 1.0,
) -> pd.DataFrame:
    metric_cols = [m for m in (metrics or DEFAULT_EVALUATION_METRICS) if m in df.columns]
    if not metric_cols:
        raise ValueError("No evaluation metric columns found in DataFrame.")

    id_vars = ["run_identifier"] if "run_identifier" in df.columns else []
    melted = df[id_vars + metric_cols].melt(
        id_vars=id_vars, var_name="metric", value_name="rate"
    )
    melted = melted.dropna(subset=["rate"])
    if max_rate is not None:
        melted = melted[melted["rate"] <= max_rate]
    if melted.empty:
        raise ValueError("No valid evaluation metric values remain after filtering.")
    return melted


def plot_evaluation_boxplot(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    max_rate: Optional[float] = 1.0,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
) -> Figure:
    """Plot boxplots for ExtractionEvaluation rates (grouped by run if multiple).

    Operation: ``VisualizationOperation.EVALUATION_BOXPLOT`` (model: ``ExtractionEvaluation``).

    Args:
        df: DataFrame produced from ``ExtractionEvaluation`` records.
        metrics: Evaluation rate columns to plot (defaults to ``groundness_rate``, ``faithfulness_rate``, ``completeness_rate``).
        max_rate: Upper threshold to filter out-of-range rates (set ``None`` to keep all values).
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["evaluation_boxplot"], "extraction_evaluation", run_identifier=[300, 301])
    """
    melted = _melt_evaluation_metrics(df, metrics=metrics, max_rate=max_rate)

    fig, ax = plt.subplots(figsize=figsize)
    has_multi_runs = (
        "run_identifier" in melted.columns
        and melted["run_identifier"].nunique(dropna=True) > 1
    )
    if has_multi_runs:
        melted["run_label"] = melted["run_identifier"].fillna("unknown").astype(str)
        sns.boxplot(data=melted, x="metric", y="rate", hue="run_label", ax=ax)
        ax.legend(title="Run Identifier")
    else:
        sns.boxplot(data=melted, x="metric", y="rate", color="steelblue", ax=ax)

    ax.set_ylim(-0.05, 1.08 if max_rate is None or max_rate <= 1.0 else max_rate * 1.05)
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylabel("Rate")
    ax.set_title(title or "Extraction Evaluation Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_evaluation_summary_range(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    max_rate: Optional[float] = 1.0,
    figsize: Tuple[float, float] = (7, 5),
    title: Optional[str] = None,
) -> Figure:
    """Plot min-max vertical range lines with mean markers for evaluation metrics.

    Operation: ``VisualizationOperation.EVALUATION_SUMMARY_RANGE`` (model: ``ExtractionEvaluation``).

    Args:
        df: DataFrame produced from ``ExtractionEvaluation`` records.
        metrics: Evaluation rate columns to summarize.
        max_rate: Upper threshold to filter out-of-range rates (set ``None`` to keep all values).
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["evaluation_summary_range"], "extraction_evaluation", run_identifier=300)
    """
    melted = _melt_evaluation_metrics(df, metrics=metrics, max_rate=max_rate)
    has_multi_runs = (
        "run_identifier" in melted.columns
        and melted["run_identifier"].nunique(dropna=True) > 1
    )

    if has_multi_runs:
        melted["label"] = (
            melted["metric"].astype(str)
            + "\n(run "
            + melted["run_identifier"].fillna("unknown").astype(str)
            + ")"
        )
    else:
        melted["label"] = melted["metric"].astype(str)

    stats = (
        melted.groupby("label", sort=False)["rate"]
        .agg(["min", "max", "mean", "count"])
        .reset_index()
    )

    palette = ["steelblue", "seagreen", "darkorange", "slateblue", "crimson", "teal"]
    fig, ax = plt.subplots(figsize=figsize)
    x_positions = list(range(1, len(stats) + 1))

    for idx, row in stats.iterrows():
        x = x_positions[idx]
        color = palette[idx % len(palette)]
        ax.vlines(x=x, ymin=row["min"], ymax=row["max"], color=color, linewidth=2, alpha=0.75)
        ax.hlines(y=row["min"], xmin=x - 0.1, xmax=x + 0.1, color=color, linewidth=2)
        ax.hlines(y=row["max"], xmin=x - 0.1, xmax=x + 0.1, color=color, linewidth=2)
        ax.scatter([x], [row["mean"]], color=color, s=90, zorder=3, marker="o")
        ax.text(
            x + 0.14,
            row["mean"],
            f"Avg: {row['mean']:.2f}",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(stats["label"].tolist(), rotation=15 if len(stats) > 3 else 0)
    ax.set_xlim(0.4, len(stats) + 0.7)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel("Rate", fontweight="bold")
    ax.set_title(
        title or f"Evaluation Summary Range ({len(df)} entries)",
        pad=12,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_evaluation_distribution(
    df: pd.DataFrame,
    metrics: Optional[Sequence[str]] = None,
    max_rate: Optional[float] = 1.0,
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot histogram of evaluation rates across metrics.

    Operation: ``VisualizationOperation.EVALUATION_DISTRIBUTION`` (model: ``ExtractionEvaluation``).

    Args:
        df: DataFrame produced from ``ExtractionEvaluation`` records.
        metrics: Evaluation rate columns to plot.
        max_rate: Upper threshold to filter out-of-range rates (set ``None`` to keep all values).
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["evaluation_distribution"], "extraction_evaluation", run_identifier=300)
    """
    melted = _melt_evaluation_metrics(df, metrics=metrics, max_rate=max_rate)
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=melted,
        x="rate",
        hue="metric",
        multiple="dodge",
        bins=10,
        shrink=0.85,
        ax=ax,
    )
    ax.set_xlabel("Rate")
    ax.set_ylabel("Count")
    ax.set_title(title or "Distribution of Evaluation Rates")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Individual plotting functions: PaperRelevance
# ---------------------------------------------------------------------------


def plot_relevance_confidence_hist(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot dodged histogram of relevance confidence scores split by `is_relevant`.

    Operation: ``VisualizationOperation.RELEVANCE_CONFIDENCE_HIST`` (model: ``PaperRelevance``).

    Args:
        df: DataFrame produced from ``PaperRelevance`` records (requires ``confidence`` and ``is_relevant``).
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["relevance_confidence_hist"], "paper_relevance", model_name="gemma-4-31b-it")
    """
    if "confidence" not in df.columns or "is_relevant" not in df.columns:
        raise ValueError("Columns 'confidence' and 'is_relevant' are required.")

    df_plot = df.dropna(subset=["confidence"]).copy()
    if df_plot.empty:
        raise ValueError("No non-null confidence values found in DataFrame.")

    df_plot["is_relevant_label"] = df_plot["is_relevant"].map(
        {True: "True", False: "False", 1: "True", 0: "False"}
    ).fillna("Unknown")

    order = [lbl for lbl in ("True", "False", "Unknown") if lbl in set(df_plot["is_relevant_label"])]

    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(
        data=df_plot,
        x="confidence",
        hue="is_relevant_label",
        hue_order=order,
        multiple="dodge",
        shrink=0.8,
        ax=ax,
    )
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Paper Count")
    ax.set_title(title or "Relevance Screening Confidence Distribution")
    fig.tight_layout()
    return fig


def plot_relevance_outcome_counts(
    df: pd.DataFrame,
    figsize: Tuple[float, float] = (7, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot counts of relevant vs non-relevant screening outcomes.

    Operation: ``VisualizationOperation.RELEVANCE_OUTCOME_COUNTS`` (model: ``PaperRelevance``).

    Args:
        df: DataFrame produced from ``PaperRelevance`` records (requires ``is_relevant``).
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["relevance_outcome_counts"], "paper_relevance", run_identifier=[400, 401])
    """
    if "is_relevant" not in df.columns:
        raise ValueError("Column 'is_relevant' is required.")

    df_plot = df.copy()
    df_plot["outcome"] = df_plot["is_relevant"].map(
        {True: "Relevant", False: "Not Relevant", 1: "Relevant", 0: "Not Relevant"}
    ).fillna("Unknown")

    fig, ax = plt.subplots(figsize=figsize)
    if "run_identifier" in df_plot.columns and df_plot["run_identifier"].nunique(dropna=True) > 1:
        df_plot["run_label"] = df_plot["run_identifier"].fillna("unknown").astype(str)
        sns.countplot(data=df_plot, x="outcome", hue="run_label", ax=ax)
        ax.legend(title="Run Identifier")
    else:
        sns.countplot(data=df_plot, x="outcome", color="steelblue", ax=ax)

    ax.set_xlabel("Screening Outcome")
    ax.set_ylabel("Paper Count")
    ax.set_title(title or "Relevance Screening Outcomes")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Individual plotting functions: PaperMetadata
# ---------------------------------------------------------------------------


def plot_paper_status_counts(
    df: pd.DataFrame,
    group_by: Optional[str] = "dataset_subset",
    figsize: Tuple[float, float] = (8, 4.5),
    title: Optional[str] = None,
) -> Figure:
    """Plot PaperMetadata counts by lifecycle status.

    Operation: ``VisualizationOperation.PAPER_STATUS_COUNTS`` (model: ``PaperMetadata``).

    Args:
        df: DataFrame produced from ``PaperMetadata`` records (requires ``status``).
        group_by: Optional column (default ``"dataset_subset"``) used to split bars by hue.
        figsize: Figure size as ``(width, height)`` in inches.
        title: Optional custom plot title.

    Usage:
        >>> visualize(["paper_status_counts"], "paper_metadata", dataset_subset="scraped")
    """
    if "status" not in df.columns:
        raise ValueError("Column 'status' is required.")

    df_plot = df.copy()
    df_plot["status"] = df_plot["status"].fillna("unknown").astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    if group_by and group_by in df_plot.columns and df_plot[group_by].nunique(dropna=True) > 1:
        df_plot[group_by] = df_plot[group_by].fillna("unknown").astype(str)
        sns.countplot(data=df_plot, x="status", hue=group_by, ax=ax)
    else:
        sns.countplot(data=df_plot, x="status", color="steelblue", ax=ax)

    ax.set_xlabel("Paper Status")
    ax.set_ylabel("Count")
    ax.set_title(title or "Paper Metadata Status Counts")
    for container in ax.containers:
        ax.bar_label(container)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Registry & unified interface
# ---------------------------------------------------------------------------

VISUALIZATION_REGISTRY: Dict[VisualizationOperation, VisualizationSpec] = {
    VisualizationOperation.PEPTIDE_COLUMN_COMPLETION: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDE_COLUMN_COMPLETION,
        supported_models=(PeptideEntry,),
        handler=plot_peptide_column_completion,
        default_filename="peptide_column_completion",
        description="Percentage of non-null/non-empty values across PeptideEntry columns.",
    ),
    VisualizationOperation.PEPTIDE_LENGTH_DISTRIBUTION: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDE_LENGTH_DISTRIBUTION,
        supported_models=(PeptideEntry,),
        handler=plot_peptide_length_distribution,
        default_filename="peptide_length_distribution",
        description="Distribution of extracted peptide sequence lengths.",
    ),
    VisualizationOperation.EXTRACTIONS_PER_PAPER: VisualizationSpec(
        operation=VisualizationOperation.EXTRACTIONS_PER_PAPER,
        supported_models=(PeptideEntry,),
        handler=plot_extractions_per_paper,
        default_filename="extractions_per_paper",
        description="Distribution of extracted peptide counts per paper.",
    ),
    VisualizationOperation.PEPTIDE_MORPHOLOGY_DISTRIBUTION: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDE_MORPHOLOGY_DISTRIBUTION,
        supported_models=(PeptideEntry,),
        handler=plot_peptide_morphology_distribution,
        default_filename="peptide_morphology_distribution",
        description="Counts of observed nanostructure morphology classes.",
    ),
    VisualizationOperation.PEPTIDE_RUN_COMPARISON: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDE_RUN_COMPARISON,
        supported_models=(PeptideEntry,),
        handler=plot_peptide_run_comparison,
        default_filename="peptide_run_comparison",
        description="Side-by-side comparison of entries, unique papers, and completion across runs.",
    ),
    VisualizationOperation.PEPTIDEMINER_CATEGORICAL_ACCURACY: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDEMINER_CATEGORICAL_ACCURACY,
        supported_models=(PeptideEntry,),
        handler=plot_peptideminer_categorical_accuracy,
        default_filename="peptideminer_categorical_accuracy",
        description="Grouped bar chart of categorical extraction accuracy (PeptideMiner Fig. 4c style).",
    ),
    VisualizationOperation.PEPTIDEMINER_NUMERICAL_MAE: VisualizationSpec(
        operation=VisualizationOperation.PEPTIDEMINER_NUMERICAL_MAE,
        supported_models=(PeptideEntry,),
        handler=plot_peptideminer_numerical_mae,
        default_filename="peptideminer_numerical_mae",
        description="Violin plot of numerical extraction MAE distributions (PeptideMiner Fig. 4d style).",
    ),
    VisualizationOperation.EVALUATION_BOXPLOT: VisualizationSpec(
        operation=VisualizationOperation.EVALUATION_BOXPLOT,
        supported_models=(ExtractionEvaluation,),
        handler=plot_evaluation_boxplot,
        default_filename="evaluation_boxplot",
        description="Boxplot of groundness, faithfulness, and completeness rates.",
    ),
    VisualizationOperation.EVALUATION_SUMMARY_RANGE: VisualizationSpec(
        operation=VisualizationOperation.EVALUATION_SUMMARY_RANGE,
        supported_models=(ExtractionEvaluation,),
        handler=plot_evaluation_summary_range,
        default_filename="evaluation_summary_range",
        description="Min-max range and average markers for evaluation metrics.",
    ),
    VisualizationOperation.EVALUATION_DISTRIBUTION: VisualizationSpec(
        operation=VisualizationOperation.EVALUATION_DISTRIBUTION,
        supported_models=(ExtractionEvaluation,),
        handler=plot_evaluation_distribution,
        default_filename="evaluation_distribution",
        description="Histogram of evaluation metric rates.",
    ),
    VisualizationOperation.RELEVANCE_CONFIDENCE_HIST: VisualizationSpec(
        operation=VisualizationOperation.RELEVANCE_CONFIDENCE_HIST,
        supported_models=(PaperRelevance,),
        handler=plot_relevance_confidence_hist,
        default_filename="relevance_confidence_hist",
        description="Histogram of relevance confidence split by relevance outcome.",
    ),
    VisualizationOperation.RELEVANCE_OUTCOME_COUNTS: VisualizationSpec(
        operation=VisualizationOperation.RELEVANCE_OUTCOME_COUNTS,
        supported_models=(PaperRelevance,),
        handler=plot_relevance_outcome_counts,
        default_filename="relevance_outcome_counts",
        description="Counts of relevant vs non-relevant screening results.",
    ),
    VisualizationOperation.PAPER_STATUS_COUNTS: VisualizationSpec(
        operation=VisualizationOperation.PAPER_STATUS_COUNTS,
        supported_models=(PaperMetadata,),
        handler=plot_paper_status_counts,
        default_filename="paper_status_counts",
        description="Counts of PaperMetadata records by lifecycle status.",
    ),
}


def _default_output_dir() -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    return project_root / "data" / "diagrams" / str(int(time.time()))


def visualize(
    operations: Union[
        VisualizationOperation,
        str,
        Sequence[Union[VisualizationOperation, str]],
    ],
    model: Optional[Union[Type[SQLModel], str]] = None,
    *,
    records: Optional[Sequence[SQLModel]] = None,
    session: Optional[Session] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    output_dir: Optional[Union[str, Path]] = None,
    save: bool = True,
    show: bool = False,
    file_format: str = "png",
    dpi: int = 150,
    plot_kwargs: Optional[Dict[str, Any]] = None,
    **db_kwargs: Any,
) -> Dict[VisualizationOperation, VisualizationResult]:
    """Unified interface to validate, query, and render visualizations.

    Args:
        operations: Single operation or sequence of operations to execute.
        model: Target SQLModel class or table name (e.g. `PeptideEntry`, `"peptide_entry"`).
            Optional if `records` is provided; if both are provided, they must match.
        records: Optional pre-fetched sequence of SQLModel records. If omitted,
            records are queried via `get_records(model, session=session, limit=limit, offset=offset, **db_kwargs)`.
        session: Optional SQLModel Session passed to `get_records`.
        limit: Optional row limit passed to `get_records`.
        offset: Row offset passed to `get_records`.
        output_dir: Directory where saved figures are written when `save=True`.
        save: Whether to save rendered figures to disk.
        show: Whether to call `plt.show()` on each figure.
        file_format: Image extension when saving (default `"png"`).
        dpi: Resolution in DPI when saving.
        plot_kwargs: Optional dict of extra keyword arguments forwarded to the plot handler.
        **db_kwargs: Column filters forwarded to `get_records` (e.g., `run_identifier=[1, 2]`).

    Returns:
        Mapping from `VisualizationOperation` to `VisualizationResult`.
    """
    if records is not None:
        record_list = list(records)
        if model is None:
            if not record_list:
                raise ValueError(
                    "`model` must be provided when `records` is empty."
                )
            resolved_model = type(record_list[0])
        else:
            resolved_model = resolve_visualization_model(model)

        for rec in record_list:
            if not isinstance(rec, resolved_model):
                raise VisualizationCompatibilityError(
                    f"Record of type '{type(rec).__name__}' does not match "
                    f"expected data type '{resolved_model.__name__}'."
                )
    else:
        if model is None:
            raise ValueError("Either `model` or `records` must be provided.")
        resolved_model = resolve_visualization_model(model)
        record_list = []  # populated after operation validation

    # Validate compatibility BEFORE querying the DB so mismatches fail fast
    resolved_ops = validate_operations_for_model(operations, resolved_model)

    if records is None:
        record_list = get_records(
            resolved_model,
            session=session,
            limit=limit,
            offset=offset,
            **db_kwargs,
        )

    if not record_list:
        logger.warning(
            f"No '{resolved_model.__name__}' records found for filters={db_kwargs}; "
            "skipping visualization."
        )
        return {}

    df = records_to_dataframe(record_list)

    target_dir: Optional[Path] = None
    if save:
        target_dir = Path(output_dir) if output_dir is not None else _default_output_dir()
        target_dir.mkdir(parents=True, exist_ok=True)

    extra_plot_kwargs = dict(plot_kwargs or {})
    results: Dict[VisualizationOperation, VisualizationResult] = {}

    for op in resolved_ops:
        spec = VISUALIZATION_REGISTRY[op]
        fig = spec.handler(df, **extra_plot_kwargs)

        out_path: Optional[Path] = None
        if save and target_dir is not None:
            out_path = target_dir / f"{spec.default_filename}.{file_format.lstrip('.')}"
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved visualization '{op.value}' to '{out_path}'")

        if show:
            plt.show()

        results[op] = VisualizationResult(
            operation=op,
            model_name=resolved_model.__name__,
            record_count=len(record_list),
            figure=fig,
            output_path=out_path,
        )

    return results
