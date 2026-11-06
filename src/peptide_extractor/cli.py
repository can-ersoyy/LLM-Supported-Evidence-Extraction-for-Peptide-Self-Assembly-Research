import click
import yaml
import logging
import os
import time

from peptide_extractor.pipeline import (
    retrive_ncbi_by_query,
    deduplicate_by_doi,
    find_abstracts_by_doi,
    remove_empty_abstract_title_papers,
    run_bulk_relevance_screening,
    fetch_paper_fulltexts,
    run_peptide_extraction,
    run_evaluation,
)
from peptide_extractor.providers import NCBIProvider, OpenAlexProvider

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@click.group()
def cli():
    """Peptide Extractor CLI"""
    pass


@cli.command(name="run_pipeline")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the configuration YAML file.",
)
def run_pipeline(config):
    """Run the peptide extraction pipeline based on a config file."""
    cfg = load_config(config)

    # Set up basic logging
    log_level = getattr(
        logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO
    )
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")

    ncbi_provider = NCBIProvider(
        api_key=ncbi_api_key, batch_size=cfg.get("ncbi_batch_size", 100)
    )
    open_alex_provider = OpenAlexProvider(api_key=open_alex_api_key)

    steps = cfg.get("steps", [])
    logger.info(f"Loaded config from '{config}'. Steps to execute: {steps}")

    if "search" in steps:
        logger.info("Running query search...")
        search_cfg = cfg.get("search", {})
        queries = search_cfg.get("queries", [])
        db = search_cfg.get("db", "pmc")
        retrive_ncbi_by_query(ncbi_provider, db=db, query_list=queries)

    if "deduplicate" in steps:
        logger.info("Deduplicating by DOI...")
        dd_cfg = cfg.get("deduplicate", {})
        normalize = dd_cfg.get("normalize", True)
        remove_empty = dd_cfg.get("remove_empty", True)
        removed = deduplicate_by_doi(normalize=normalize, remove_empty=remove_empty)
        logger.info(f"Removed {removed} papers with duplicate doi")

    if "find_abstracts" in steps:
        logger.info("Finding missing abstracts...")
        find_abstracts_by_doi(open_alex_api_key)

    if "clean_empty" in steps:
        logger.info("Removing papers with empty abstract or title...")
        removed = remove_empty_abstract_title_papers()
        logger.info(f"Removed {removed} papers with empty abstract or title")

    if "relevance" in steps:
        logger.info("Running relevance screening...")
        rel_cfg = cfg.get("relevance", {})
        start_time = time.time()
        success, fails, identifier = run_bulk_relevance_screening(
            model_name=rel_cfg.get("model_name", "gemini-3.5-flash-lite"),
            provider=rel_cfg.get("provider", "google"),
            delay_seconds=rel_cfg.get("delay_seconds", 3),
            batch_size=rel_cfg.get("batch_size", 50),
            **rel_cfg.get("kwargs", {}),
        )
        end_time = time.time()
        logger.info(
            f"Relevance Success: {success}, Fails: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
        )

    if "fetch_fulltext" in steps:
        logger.info("Fetching fulltexts...")
        ft_cfg = cfg.get("fetch_fulltext", {})
        fetch_paper_fulltexts(
            ncbi_provider=ncbi_provider,
            open_alex_provider=open_alex_provider,
            **ft_cfg.get("kwargs", {}),
        )

    if "extraction" in steps:
        logger.info("Running extraction...")
        ext_cfg = cfg.get("extraction", {})
        start_time = time.time()
        fails, identifier = run_peptide_extraction(
            provider=ext_cfg.get("provider", "google"),
            model_name=ext_cfg.get("model_name", "gemini-3.5-flash-lite"),
            delay_seconds=ext_cfg.get("delay_seconds", 4),
            **ext_cfg.get("kwargs", {}),
        )
        end_time = time.time()
        logger.info(
            f"Extraction failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
        )

    if "evaluation" in steps:
        logger.info("Running evaluation...")
        eval_cfg = cfg.get("evaluation", {})
        start_time = time.time()
        fails, identifier = run_evaluation(**eval_cfg.get("kwargs", {}))
        end_time = time.time()
        logger.info(
            f"Evaluation failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
        )


if __name__ == "__main__":
    cli()
