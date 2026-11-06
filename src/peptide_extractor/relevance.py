from peptide_extractor.schemas import BulkRelevanceResults
from peptide_extractor.llm import LLMController
from peptide_extractor.schemas import RelevanceResult
import logging
import time
from typing import Any, List, Optional
from dotenv import find_dotenv, load_dotenv
from sqlmodel import Session

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv(find_dotenv(usecwd=True))


from peptide_extractor.db.paper_metadata import (
    PaperMetadata,
    PaperStatus,
    get_paper_metadatas,
    update_paper_status,
)
from peptide_extractor.db.paper_relevance import save_paper_relevance

SYSTEM_PROMPT_V1 = """You are an expert in scientific paper reading and computational biology.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on 4-20mer peptide self-assembly research based only on its title and abstract.

You must determine if the paper reports experimental self-assembly or nanostructure formation of short peptides (e.g., nanofibers, hydrogels, nanotubes, vesicles, organogels, nanospheres, amyloid fibrils) along with experimental conditions (e.g., pH, solvent, concentration, temperature, detection methods).

Be careful not to discard papers that do not explicitly state full experimental details in the abstract but hint at experimental work. It is better to include a borderline paper (marking as relevant with lower confidence) than to mistakenly discard a paper containing experimental data.

Provide structured output containing:
- detected_clues: List of specific techniques, peptide identities, nanostructures, or experimental conditions mentioned.
- explanation: Clear justification for the relevance assessment.
- is_relevant: True if the paper likely contains experimental peptide self-assembly data in its full text, False otherwise.
- confidence: Floating-point confidence score between 0.0 and 1.0.
"""

SYSTEM_PROMPT_V2 = """You are an expert in scientific paper reading and peptide research.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on 4-20mer peptide self-assembly/amyloid research based only on its title and abstract. Based on your decision, fulltext of the paper will be retrieved.

You must determine if the fulltext of the paper likely reports experimental self-assembly or nanostructure formation of short peptides (e.g., nanofibers, hydrogels, nanotubes, vesicles, organogels, nanospheres, amyloid fibrils ...) along with experimental conditions (e.g., pH, solvent, concentration, temperature, detection methods ...).

Keep in mind, most of the time for longer peptides, oligopeptide naming system (e.g. dodecapeptide for 12-mer) is not provided and peptides might have special names.

Be careful not to discard papers that do not explicitly state full experimental details in the abstract but hint at experimental work. It is better to include a borderline paper (marking as relevant with lower confidence) than to mistakenly discard a paper containing experimental data.

Provide structured output containing:
- detected_clues: List of specific techniques, peptide identities, nanostructures, or experimental conditions mentioned.
- explanation: Clear justification for the relevance assessment.
- is_relevant: True if the paper likely contains experimental peptide self-assembly data in its full text, False otherwise.
- confidence: Floating-point confidence score between 0.0 and 1.0.
"""

SYSTEM_PROMPT_V3 = """You are an expert in scientific paper reading and biophysics specializing in peptide self-assembly and amyloid aggregation.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on 4-20mer peptide self-assembly research based only on its title and abstract.

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

SYSTEM_PROMPT_GOLDEN_BENCHMARK = """You are an expert in scientific paper reading and computational biology.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on peptide self-assembly/amyloid research based only on its title and abstract.

You must determine if the paper reports experimental self-assembly or nanostructure formation of short peptides (e.g., nanofibers, hydrogels, nanotubes, vesicles, organogels, nanospheres, amyloid fibrils) along with experimental conditions (e.g., pH, solvent, concentration, temperature, detection methods).

Be careful not to discard papers that do not explicitly state full experimental details in the abstract but hint at experimental work. It is better to include a borderline paper (marking as relevant with lower confidence) than to mistakenly discard a paper containing experimental data.

Provide structured output containing:
- detected_clues: List of specific techniques, peptide identities, nanostructures, or experimental conditions mentioned.
- explanation: Clear justification for the relevance assessment.
- is_relevant: True if the paper likely contains experimental peptide self-assembly data in its full text, False otherwise.
- confidence: Floating-point confidence score between 0.0 and 1.0.
"""

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
    **kwargs: Any,
) -> tuple[List[tuple[PaperMetadata, RelevanceResult]], List[int]]:
    """Screen papers given conditions for relevance."""

    if classifier is None:
        classifier = LLMController(
            model_name=model_name,
            provider=provider,
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_V3,
            # base_url="http://localhost:11435"
        )

    papers_to_screen = get_paper_metadatas(limit=batch_size, session=session, **kwargs)

    results: List[tuple[PaperMetadata, RelevanceResult]] = []
    failed_paper_ids = []

    for paper in papers_to_screen:
        paper_id = paper.id
        try:
            relevance_result: RelevanceResult = classifier.classify_paper(
                paper.title, paper.abstract
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
            logger.info(f"Paper {paper_id} processed")
            logger.info(f"Paper with id {paper_id} is processed")

        except Exception as e:
            if session is not None:
                session.rollback()
            failed_paper_ids.append(paper_id)

            logger.info(f"Skipping paper ID {paper_id} due to error: {e}")

            continue
        finally:
            time.sleep(delay_seconds)

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
            logger.info(prepared_batch)

            relevance_results: BulkRelevanceResults = classifier.bulk_classify_paper(
                prepared_batch
            )

            for result in relevance_results.results:
                id = result.paper_id
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
                    logger.info(f"Paper {id} processed")
                    logger.info(f"Paper with id {id} is processed")
                except Exception as e:
                    if session is not None:
                        session.rollback()
                    failed_paper_ids.append(id)
                    logger.info(f"Skipping paper {id} due to error: {e}")
                    continue

        except Exception as e:
            if session is not None:
                session.rollback()
                # failed_paper_ids.append(result.id)

                logger.info(
                    f"Skipping papers {current_batch[0].id} - {current_batch[-1].id} due to error: {e}"
                )

                continue
        finally:
            time.sleep(delay_seconds)

    return results, failed_paper_ids
