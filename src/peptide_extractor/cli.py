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
from peptide_extractor.logging_utils import step_logger

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
@click.option(
    "--log-to-file/--no-log-to-file",
    default=None,
    help="Store logs for each step in docs/logs/[step]/{run_identifier}.log",
)
def run_pipeline(config, log_to_file):
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

    # Determine whether file logging is enabled globally
    if log_to_file is None:
        global_log_to_file = bool(
            cfg.get("log_to_file", cfg.get("save_logs", cfg.get("file_logging", False)))
        )
    else:
        global_log_to_file = bool(log_to_file)

    def is_step_log_enabled(step_name: str) -> bool:
        step_cfg = cfg.get(step_name, {})
        if isinstance(step_cfg, dict):
            for k in ("log_to_file", "save_logs", "file_logging"):
                if k in step_cfg:
                    return bool(step_cfg[k])
        return global_log_to_file

    ncbi_api_key = os.getenv("NCBI_API_KEY")
    open_alex_api_key = os.getenv("OPEN_ALEX_API_KEY")

    ncbi_provider = NCBIProvider(
        api_key=ncbi_api_key, batch_size=cfg.get("ncbi_batch_size", 100)
    )
    open_alex_provider = OpenAlexProvider(api_key=open_alex_api_key)

    steps = cfg.get("steps", [])
    logger.info(f"Loaded config from '{config}'. Steps to execute: {steps}")

    if "search" in steps:
        search_cfg = cfg.get("search", {})
        run_identifier = search_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("search")
        with step_logger("search", run_identifier, enabled=step_enabled):
            logger.info("Running query search...")
            queries = search_cfg.get("queries", [])
            db = search_cfg.get("db", "pmc")
            retrive_ncbi_by_query(
                ncbi_provider,
                db=db,
                query_list=queries,
                run_identifier=run_identifier,
                log_to_file=step_enabled,
            )

    if "deduplicate" in steps:
        dd_cfg = cfg.get("deduplicate", {})
        run_identifier = dd_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("deduplicate")
        with step_logger("deduplicate", run_identifier, enabled=step_enabled):
            logger.info("Deduplicating by DOI...")
            normalize = dd_cfg.get("normalize", True)
            remove_empty = dd_cfg.get("remove_empty", True)
            removed = deduplicate_by_doi(normalize=normalize, remove_empty=remove_empty)
            logger.info(f"Removed {removed} papers with duplicate doi")

    if "find_abstracts" in steps:
        fa_cfg = cfg.get("find_abstracts", {})
        run_identifier = fa_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("find_abstracts")
        with step_logger("find_abstracts", run_identifier, enabled=step_enabled):
            logger.info("Finding missing abstracts...")
            find_abstracts_by_doi(open_alex_api_key)

    if "clean_empty" in steps:
        ce_cfg = cfg.get("clean_empty", {})
        run_identifier = ce_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("clean_empty")
        with step_logger("clean_empty", run_identifier, enabled=step_enabled):
            logger.info("Removing papers with empty abstract or title...")
            removed = remove_empty_abstract_title_papers()
            logger.info(f"Removed {removed} papers with empty abstract or title")

    if "relevance" in steps:
        rel_cfg = cfg.get("relevance", {})
        run_identifier = rel_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("relevance")
        with step_logger("relevance", run_identifier, enabled=step_enabled):
            logger.info("Running relevance screening...")
            start_time = time.time()
            success, fails, identifier = run_bulk_relevance_screening(
                model_name=rel_cfg.get("model_name", "gemini-3.5-flash-lite"),
                provider=rel_cfg.get("provider", "google"),
                delay_seconds=rel_cfg.get("delay_seconds", 3),
                batch_size=rel_cfg.get("batch_size", 50),
                run_identifier=run_identifier,
                log_to_file=step_enabled,
                **rel_cfg.get("kwargs", {}),
            )
            end_time = time.time()
            logger.info(
                f"Relevance Success: {success}, Fails: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
            )

    if "fetch_fulltext" in steps or "fetch" in steps:
        ft_cfg = cfg.get("fetch") or cfg.get("fetch_fulltext", {})
        run_identifier = ft_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("fetch") or is_step_log_enabled("fetch_fulltext")
        with step_logger("fetch", run_identifier, enabled=step_enabled):
            logger.info("Fetching fulltexts...")
            success, fails, identifier = fetch_paper_fulltexts(
                ncbi_provider=ncbi_provider,
                open_alex_provider=open_alex_provider,
                run_identifier=run_identifier,
                log_to_file=step_enabled,
                **ft_cfg.get("kwargs", {}),
            )
            logger.info(
                f"Fetch Fulltext Success: {success}, Fails: {fails}, Identifier: {identifier}"
            )

    if "extraction" in steps:
        ext_cfg = cfg.get("extraction", {})
        run_identifier = ext_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("extraction")
        with step_logger("extraction", run_identifier, enabled=step_enabled):
            logger.info("Running extraction...")
            start_time = time.time()
            fails, identifier = run_peptide_extraction(
                provider=ext_cfg.get("provider", "google"),
                model_name=ext_cfg.get("model_name", "gemini-3.5-flash-lite"),
                delay_seconds=ext_cfg.get("delay_seconds", 4),
                run_identifier=run_identifier,
                log_to_file=step_enabled,
                **ext_cfg.get("kwargs", {}),
            )
            end_time = time.time()
            logger.info(
                f"Extraction failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
            )

    if "evaluation" in steps:
        eval_cfg = cfg.get("evaluation", {})
        run_identifier = eval_cfg.get("run_identifier") or int(time.time())
        step_enabled = is_step_log_enabled("evaluation")
        with step_logger("evaluation", run_identifier, enabled=step_enabled):
            logger.info("Running evaluation...")
            start_time = time.time()
            fails, identifier = run_evaluation(
                run_identifier=run_identifier,
                log_to_file=step_enabled,
                **eval_cfg.get("kwargs", {}),
            )
            end_time = time.time()
            logger.info(
                f"Evaluation failed for papers with IDs: {fails}, Identifier: {identifier}, Time: {end_time - start_time}"
            )


if __name__ == "__main__":
    cli()
