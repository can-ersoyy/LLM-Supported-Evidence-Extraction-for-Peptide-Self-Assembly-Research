import click
import yaml
import logging
import os
import time

from pathlib import Path
from peptide_extractor.pipeline import (
    ALL_PIPELINE_STEPS,
    retrive_ncbi_by_query,
    deduplicate_by_doi,
    find_abstracts_by_doi,
    remove_empty_abstract_title_papers,
    run_relevance_screening,
    run_bulk_relevance_screening,
    fetch_paper_fulltexts,
    run_peptide_extraction,
    run_evaluation,
    run_visualization,
    run_full_pipeline,
)
from peptide_extractor.logging_utils import step_logger
from peptide_extractor.visualization import VisualizationOperation

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


@click.group()
def cli():
    """Peptide Extractor CLI"""
    pass


@cli.command(name="run_pipeline")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=False,
    default=None,
    help="Path to the configuration YAML file. Optional when passing CLI flags directly.",
)
@click.option(
    "-q",
    "--query",
    "queries",
    multiple=True,
    help="PubMed/PMC search query (can be specified multiple times).",
)
@click.option(
    "--db",
    default=None,
    help="NCBI database for search (e.g. 'pmc' or 'pubmed').",
)
@click.option(
    "--pmid",
    "pmids",
    multiple=True,
    help="Target PMID(s) to process through the pipeline.",
)
@click.option(
    "--paper-id",
    "paper_ids",
    multiple=True,
    type=int,
    help="Target PaperMetadata ID(s) to process through the pipeline.",
)
@click.option(
    "--step",
    "steps",
    multiple=True,
    help=f"Step(s) to run (default: all). Choices: {', '.join(ALL_PIPELINE_STEPS)}, all.",
)
@click.option(
    "--provider",
    default=None,
    help="Default LLM provider for relevance and extraction (e.g. 'google', 'ollama').",
)
@click.option(
    "--model",
    "model_name",
    default=None,
    help="Default LLM model name for relevance and extraction.",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Maximum number of papers to process when no explicit IDs are given.",
)
@click.option(
    "--run-id",
    "run_identifier",
    type=int,
    default=None,
    help="Shared run_identifier for the pipeline execution.",
)
@click.option(
    "--log-to-file/--no-log-to-file",
    default=None,
    help="Store logs for each step in docs/logs/[step]/{run_identifier}.log",
)
def run_pipeline(
    config,
    queries,
    db,
    pmids,
    paper_ids,
    steps,
    provider,
    model_name,
    limit,
    run_identifier,
    log_to_file,
):
    """Run the end-to-end peptide extraction pipeline.

    Steps automatically pass their outputs (discovered/usable paper IDs and
    run_identifier) to downstream steps, so you do not need to supply inputs
    separately for each step.
    """
    if config is not None:
        cfg = load_config(config)
    elif not any([queries, pmids, paper_ids, steps, provider, model_name, limit, run_identifier]):
        default_cfg = Path("default_config.yaml")
        cfg = load_config(default_cfg) if default_cfg.exists() else {}
    else:
        cfg = {}

    if queries:
        cfg["queries"] = list(queries)
    if db is not None:
        cfg["db"] = db
    if pmids:
        cfg["pmids"] = list(pmids)
    if paper_ids:
        cfg["paper_ids"] = list(paper_ids)
    if steps:
        cfg["steps"] = list(steps)
    if provider is not None:
        cfg["provider"] = provider
    if model_name is not None:
        cfg["model_name"] = model_name
    if limit is not None:
        cfg["limit"] = limit
    if run_identifier is not None:
        cfg["run_identifier"] = run_identifier

    # Set up basic logging
    log_level = getattr(
        logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    logger.info(
        f"Loaded config from '{config or 'CLI/defaults'}'. Steps to execute: {cfg.get('steps', 'all')}"
    )

    result = run_full_pipeline(
        config=cfg,
        log_to_file=log_to_file,
        _step_fns={
            "search": retrive_ncbi_by_query,
            "deduplicate": deduplicate_by_doi,
            "find_abstracts": find_abstracts_by_doi,
            "clean_empty": remove_empty_abstract_title_papers,
            "relevance_bulk": run_bulk_relevance_screening,
            "relevance_single": run_relevance_screening,
            "fetch_fulltext": fetch_paper_fulltexts,
            "extraction": run_peptide_extraction,
            "evaluation": run_evaluation,
            "visualization": run_visualization,
        },
    )
    click.echo(result.summary())


@cli.command(name="visualize")
@click.option(
    "-m",
    "--model",
    required=True,
    help="Target DB table/model name (e.g. peptide_entry, extraction_evaluation, paper_relevance, paper_metadata).",
)
@click.option(
    "-o",
    "--operation",
    "operations",
    multiple=True,
    required=True,
    type=click.Choice([op.value for op in VisualizationOperation], case_sensitive=False),
    help="Visualization operation(s) to run. Can be specified multiple times.",
)
@click.option(
    "--run-id",
    "run_ids",
    multiple=True,
    type=int,
    help="Optional run_identifier(s) to filter records.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help="Directory to save generated diagrams.",
)
@click.option(
    "--format",
    "file_format",
    default="png",
    show_default=True,
    help="Output image format.",
)
@click.option("--show/--no-show", default=False, help="Display plots interactively.")
def visualize_cmd(model, operations, run_ids, output_dir, file_format, show):
    """Generate visualizations from database records."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    db_kwargs = {}
    if run_ids:
        db_kwargs["run_identifier"] = list(run_ids) if len(run_ids) > 1 else run_ids[0]

    results, identifier = run_visualization(
        operations=list(operations),
        model=model,
        output_dir=output_dir,
        save=True,
        show=show,
        file_format=file_format,
        **db_kwargs,
    )
    for op, res in results.items():
        click.echo(f"[{op.value}] records={res.record_count} -> {res.output_path}")


if __name__ == "__main__":
    cli()
