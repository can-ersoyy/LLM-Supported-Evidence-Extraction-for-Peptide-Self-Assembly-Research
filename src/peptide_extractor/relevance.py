from peptide_extractor.env import load_env
from peptide_extractor.schemas import BulkRelevanceResults
from peptide_extractor.llm import LLMController
from peptide_extractor.schemas import RelevanceResult
import logging
import time
from typing import Any, List, Optional
from sqlmodel import Session

logger = logging.getLogger(__name__)

load_env()

from peptide_extractor.db.paper_metadata import (
    PaperMetadata,
    PaperStatus,
    get_paper_metadatas,
    update_paper_status,
)
from peptide_extractor.db.paper_relevance import save_paper_relevance
from peptide_extractor.utils import retry_on_external_error


SYSTEM_PROMPT_BULK = """
You are an expert in scientific paper reading and biophysics specializing in peptide self-assembly and amyloid aggregation.
Your task is to determine whether a batch of scientific papers are relevant for an information extraction pipeline on 4-20mer peptide self-assembly research based only on their titles and abstracts.

The project extracts experimental records for short peptides (roughly 4-20 residues, including canonical sequences, capped/modified peptides, and amyloidogenic fragments), capturing self-assembly or phase behavior (e.g., nanofibers, hydrogels, nanotubes, nanospheres, amyloid fibrils, ribbons, vesicles, or amorphous aggregates) and associated experimental conditions (e.g., pH, concentration, solvent/buffer, temperature, characterization methods).

Because abstracts are dense summaries that often omit detailed experimental parameters:
- Assume a paper is RELEVANT (`is_relevant: True`) if it reports or strongly implies original experimental (wet-lab) work on peptide self-assembly, aggregation, or amyloid formation.
- Do NOT require full quantitative experimental conditions (such as exact pH or concentration values) or full sequence lengths to be explicitly stated in the abstract; hints of physical characterization (e.g., microscopy, spectroscopy, gelation, kinetic assays) are sufficient.
- Only mark `is_relevant: False` if the paper is unambiguously out of scope: purely theoretical/computational studies without experimental validation, general review articles/editorials, or studies exclusively on intact, full-length proteins without isolated peptide fragments.
- When in doubt, prioritize recall and classify borderline papers as relevant with a lower confidence score.

Provide structured output containing:
- detected_clues: List of specific techniques, peptide identities, nanostructures, or experimental conditions mentioned.
- explanation: Clear justification for the relevance assessment.
- is_relevant: True if the paper likely contains experimental peptide self-assembly data in its full text, False otherwise.
- confidence: Floating-point confidence score between 0.0 and 1.0.
"""


def screen_papers(
    batch_size: Optional[int] = None,
    classifier: Optional[LLMController] = None,
    model_name: str = "gemini-3.5-flash-lite",
    provider: str = "google",
    session: Optional[Session] = None,
    delay_seconds: float = 2.0,
    run_identifier: int = 0,
    update_status: bool = True,
    max_retries: int = 0,
    retry_delay: float = 2.0,
    **kwargs: Any,
) -> tuple[List[tuple[PaperMetadata, RelevanceResult]], List[int]]:
    """Screen papers given conditions for relevance."""

    if classifier is None:
        classifier = LLMController(
            model_name=model_name,
            provider=provider,
            structured_output=RelevanceResult,
            # system_prompt=SYSTEM_PROMPT_V3,
            # base_url="http://localhost:11435"
        )

    papers_to_screen = get_paper_metadatas(limit=batch_size, session=session, **kwargs)

    results: List[tuple[PaperMetadata, RelevanceResult]] = []
    failed_paper_ids = []

    for paper in papers_to_screen:
        paper_id = paper.id
        try:
            relevance_result: RelevanceResult = retry_on_external_error(
                classifier.classify_paper,
                paper.title,
                paper.abstract,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context=f"relevance paper {paper_id}",
            )

            import json

            clues_str = json.dumps(relevance_result.detected_clues)

            save_paper_relevance(
                paper_id=paper_id,
                detected_clues=clues_str,
                explanation=relevance_result.explanation,
                is_relevant=relevance_result.is_relevant,
                confidence=relevance_result.confidence,
                provider=provider,
                model_name=model_name,
                run_identifier=run_identifier,
                session=session,
            )

            if update_status:
                new_status = (
                    PaperStatus.USABLE
                    if relevance_result.is_relevant
                    else PaperStatus.DISCARD
                )
                update_paper_status(paper_id, new_status, session=session)

            results.append((paper, relevance_result))
            logger.info(
                f"Paper {paper_id}: relevant={relevance_result.is_relevant}, confidence={relevance_result.confidence}"
            )

        except Exception as e:
            if session is not None:
                session.rollback()
            failed_paper_ids.append(paper_id)

            logger.error(f"Skipping paper ID {paper_id} due to error: {e}")

            continue
        finally:
            time.sleep(delay_seconds)

    if failed_paper_ids:
        logger.warning(
            f"Relevance screening failed for {len(failed_paper_ids)} paper(s) with IDs: {failed_paper_ids}"
        )
    else:
        logger.info("Relevance screening completed with 0 failed papers.")

    return results, failed_paper_ids


def bulk_screen_papers(
    batch_size: Optional[int] = 10,
    classifier: Optional[LLMController] = None,
    model_name: str = "gemini-3.5-flash-lite",
    provider: str = "google",
    session: Optional[Session] = None,
    delay_seconds: float = 2.0,
    run_identifier: int = 0,
    update_status: bool = True,
    max_retries: int = 0,
    retry_delay: float = 2.0,
    **kwargs: Any,
):

    if classifier is None:
        classifier = LLMController(
            model_name=model_name,
            provider=provider,
            structured_output=BulkRelevanceResults,
            system_prompt=SYSTEM_PROMPT_BULK,
            # base_url="https://breaking-trees-favorites-replied.trycloudflare.com",
            base_url="http://localhost:11435",
        )

    results: List[tuple[PaperMetadata, RelevanceResult]] = []
    failed_paper_ids = []

    papers_to_screen = get_paper_metadatas(session=session, **kwargs)

    for papers in range(0, len(papers_to_screen), batch_size):
        try:
            current_batch = papers_to_screen[papers : papers + batch_size]
            prepared_batch = [
                (paper.id, paper.title, paper.abstract) for paper in current_batch
            ]
            logger.info(
                f"Screening relevance batch of {len(current_batch)} paper(s): {[p.id for p in current_batch]}"
            )

            relevance_results: BulkRelevanceResults = retry_on_external_error(
                classifier.bulk_classify_paper,
                prepared_batch,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context=f"bulk relevance batch {current_batch[0].id}-{current_batch[-1].id}",
            )

            returned_ids = set()
            for result in relevance_results.results:
                id = result.paper_id
                returned_ids.add(id)
                real_result = result.relevance
                try:
                    import json

                    clues_str = json.dumps(real_result.detected_clues)

                    save_paper_relevance(
                        paper_id=id,
                        detected_clues=clues_str,
                        explanation=real_result.explanation,
                        is_relevant=real_result.is_relevant,
                        confidence=real_result.confidence,
                        provider=provider,
                        model_name=model_name,
                        run_identifier=run_identifier,
                        session=session,
                    )

                    if update_status:
                        new_status = (
                            PaperStatus.USABLE
                            if real_result.is_relevant
                            else PaperStatus.DISCARD
                        )
                        update_paper_status(id, new_status, session=session)

                    results.append((id, real_result))
                    logger.info(
                        f"Paper {id}: relevant={real_result.is_relevant}, confidence={real_result.confidence}"
                    )
                except Exception as e:
                    if session is not None:
                        session.rollback()
                    failed_paper_ids.append(id)
                    logger.error(f"Skipping paper {id} due to error: {e}")
                    continue

            missing_ids = [p.id for p in current_batch if p.id not in returned_ids]
            if missing_ids:
                failed_paper_ids.extend(missing_ids)
                logger.warning(
                    f"Bulk relevance response omitted paper IDs {missing_ids}; recorded as failed."
                )

        except Exception as e:
            if session is not None:
                session.rollback()
            batch_ids = [paper.id for paper in current_batch]
            failed_paper_ids.extend(batch_ids)
            logger.error(
                f"Skipping papers {batch_ids} due to error: {e}"
            )
            continue
        finally:
            time.sleep(delay_seconds)

    if failed_paper_ids:
        logger.warning(
            f"Bulk relevance screening failed for {len(failed_paper_ids)} paper(s) with IDs: {failed_paper_ids}"
        )
    else:
        logger.info("Bulk relevance screening completed with 0 failed papers.")

    return results, failed_paper_ids
