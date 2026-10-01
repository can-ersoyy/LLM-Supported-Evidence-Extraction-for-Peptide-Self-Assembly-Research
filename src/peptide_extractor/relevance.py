from peptide_extractor.schemas import RelevanceResult
from peptide_extractor.db import DatasetSubset
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from sqlmodel import Session

logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv(find_dotenv(usecwd=True))

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from peptide_extractor.db.paper_metadata import (
    PaperMetadata,
    PaperStatus,
    get_paper_metadatas,
    update_paper_status,
)
from peptide_extractor.db.paper_relevance import PaperRelevance, save_paper_relevance

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

SYSTEM_PROMPT_GOLDEN_BENCHMARK= """You are an expert in scientific paper reading and computational biology.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on peptide self-assembly/amyloid research based only on its title and abstract.

You must determine if the paper reports experimental self-assembly or nanostructure formation of short peptides (e.g., nanofibers, hydrogels, nanotubes, vesicles, organogels, nanospheres, amyloid fibrils) along with experimental conditions (e.g., pH, solvent, concentration, temperature, detection methods).

Be careful not to discard papers that do not explicitly state full experimental details in the abstract but hint at experimental work. It is better to include a borderline paper (marking as relevant with lower confidence) than to mistakenly discard a paper containing experimental data.

Provide structured output containing:
- detected_clues: List of specific techniques, peptide identities, nanostructures, or experimental conditions mentioned.
- explanation: Clear justification for the relevance assessment.
- is_relevant: True if the paper likely contains experimental peptide self-assembly data in its full text, False otherwise.
- confidence: Floating-point confidence score between 0.0 and 1.0.
"""


USER_PROMPT = """Here is the title and abstract of a paper:

Title: {title}

Abstract: {abstract}
"""





from peptide_extractor.llm import LLMController, get_chat_model


def evaluate_paper_relevance(
    paper: PaperMetadata,
    classifier: Optional[LLMController] = None,
    session: Optional[Session] = None,
    update_status: bool = True,
    run_identifier: int = 0,
) -> tuple[PaperRelevance, RelevanceResult]:
    """Evaluate relevance for a single PaperMetadata record.

    Runs LLM classification, inserts a new PaperRelevance record into the database
    (supporting multiple evaluation runs per paper metadata object), and optionally
    updates paper.status to USABLE or DISCARD.
    """
    if classifier is None:
        classifier = LLMController(
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        )

    if paper.id is None:
        raise ValueError("PaperMetadata object must have a non-None primary key id.")

    rel_result: RelevanceResult = classifier.classify_paper(paper)

    import json
    clues_str = json.dumps(rel_result.detected_clues) if isinstance(rel_result.detected_clues, list) else str(rel_result.detected_clues)
    raw_provider = getattr(classifier, "provider", None)
    provider_name = raw_provider if isinstance(raw_provider, str) and raw_provider.strip() else "google_genai"
    raw_model = getattr(classifier, "model_name", None)
    model_name_val = raw_model if isinstance(raw_model, str) and raw_model.strip() else "gemma-4-31b-it"

    relevance_record = save_paper_relevance(
        paper_id=paper.id,
        detected_clues=clues_str,
        explanation=rel_result.explanation,
        is_relevant=rel_result.is_relevant,
        confidence=rel_result.confidence,
        provider=provider_name,
        model_name=model_name_val,
        run_identifier=run_identifier,
        session=session,
    )

    if update_status:
        new_status = PaperStatus.USABLE if rel_result.is_relevant else PaperStatus.DISCARD
        update_paper_status(paper.id, new_status, session=session)

    return relevance_record, rel_result


def screen_papers(
    batch_size: Optional[int] = None,
    classifier: Optional[LLMController] = None,
    session: Optional[Session] = None,
    delay_seconds: float = 2.0,
    golden_dataset: bool = False,
    **kwargs: Any,
) -> tuple[List[tuple[PaperMetadata, RelevanceResult]], List[int], int]:
    """Screen papers given conditions for relevance.
    """
    if classifier is None:
        classifier = LLMController(
            structured_output=RelevanceResult,
            system_prompt=SYSTEM_PROMPT_GOLDEN_BENCHMARK,
        )



    if golden_dataset:
        papers_to_screen = get_paper_metadatas(limit=batch_size, session=session, dataset_subset=DatasetSubset.GOLDEN_BENCHMARK)
    else:
        papers_to_screen = get_paper_metadatas(limit=batch_size, session=session, **kwargs)

    results: List[tuple[PaperMetadata, RelevanceResult]] = []
    failed_paper_ids = []
    run_identifier = int(time.time())

    for paper in papers_to_screen:
        if paper.id is None:
            continue
        paper_id = paper.id
        try:
            relevance_rec, rel_result = evaluate_paper_relevance(
                paper=paper,
                classifier=classifier,
                session=session,
                update_status=True,
                run_identifier=run_identifier,
            )
            results.append((paper, rel_result))
            print(f"Paper {paper_id} processed")
            logger.info(f"Paper with id {paper_id} is processed")
        except Exception as e:
            if session is not None:
                session.rollback()
            failed_paper_ids.append(paper_id)
            print(f"Skipping paper ID {paper_id} due to API evaluation error: {e}")
            continue
        finally:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return results, failed_paper_ids, run_identifier