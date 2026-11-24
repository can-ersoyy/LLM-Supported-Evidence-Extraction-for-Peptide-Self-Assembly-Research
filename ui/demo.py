"""Isolated, deterministic sandbox demo of the full peptide_extractor pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from sqlmodel import select

from peptide_extractor.db import (
    DatasetSubset,
    ExtractionEvaluation,
    ExtractionMode,
    PaperContent,
    PaperMetadata,
    PaperRelevance,
    PaperStatus,
    PeptideEntry,
    get_engine,
    get_session_context,
    init_db,
)
from ui.backend import job_manager

logger = logging.getLogger("peptide_extractor.demo")

DEMO_RUN_ID = 9001


def init_demo_db(db_path: str) -> None:
    """Create/initialize an empty demo SQLite database at db_path."""
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    init_db(engine)


def reset_demo_db(db_path: str) -> None:
    """Wipe and re-initialize the demo SQLite database."""
    p = Path(db_path)
    if p.exists():
        p.unlink()
    init_demo_db(db_path)


def _demo_search(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    init_db(engine)
    logger.info("Querying NCBI PMC (demo): peptide self-assembly...")
    time.sleep(0.3)

    sample_papers: List[PaperMetadata] = [
        PaperMetadata(
            doi="10.1002/adma.200600881",
            pmid="18052148",
            pmcid="PMC2791716",
            title="Self-assembly of Fmoc-diphenylalanine into fibrous hydrogels",
            abstract=(
                "Fmoc-FF dipeptide self-assembles in aqueous PBS (pH 7.4) at 25 °C "
                "into entangled nanofibrillar hydrogels characterized by TEM, AFM, and rheology."
            ),
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.GOLDEN_BENCHMARK,
            source_databases="SAPDB; PeptideMiner",
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
        PaperMetadata(
            doi="10.1073/pnas.96.18.10001",
            pmid="10468553",
            title="Amyloid fibril formation by the Alzheimer's peptide fragment KLVFF",
            abstract=(
                "The pentapeptide KLVFF (residues 16-20 of Aβ) forms cross-beta amyloid "
                "fibers at pH 7.0 and 37 °C, verified by ThT fluorescence, CD, and TEM."
            ),
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            source_databases="NCBI_SEARCH",
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
        PaperMetadata(
            doi="10.1021/nn4001234",
            pmid="23456789",
            title="Surfactant-like peptide Ac-IIIK-NH2 forms well-defined nanotubes",
            abstract=(
                "Ac-IIIK-NH2 (2.0 mg/ml in H2O, pH 6.8, 25 °C) self-assembles into "
                "uniform hollow nanotubes observed by Cryo-EM and AFM."
            ),
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            source_databases="NCBI_SEARCH",
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
        # Duplicate DOI to demonstrate deduplication
        PaperMetadata(
            doi="10.1002/adma.200600881",
            pmid="18052148",
            title="Self-assembly of Fmoc-diphenylalanine into fibrous hydrogels (Duplicate)",
            abstract="Duplicate record from secondary query.",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
        # Empty abstract to demonstrate cleaning
        PaperMetadata(
            doi="10.1000/empty-abstract",
            pmid="9990001",
            title="Conference abstract without body text",
            abstract="",
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
        # Non-experimental review/computational paper to demonstrate DISCARD
        PaperMetadata(
            doi="10.1016/j.comp.2023.001",
            pmid="37000111",
            title="Molecular dynamics simulation review of peptide aggregation",
            abstract=(
                "This review summarizes coarse-grained molecular dynamics force fields "
                "for peptide aggregation without reporting new wet-lab experiments."
            ),
            status=PaperStatus.NOT_PROCESSED,
            is_open_access=True,
            dataset_subset=DatasetSubset.SCRAPED,
            provider="ncbi",
            run_identifier=DEMO_RUN_ID,
        ),
    ]

    with get_session_context(db_engine=engine) as session:
        existing = session.exec(select(PaperMetadata)).all()
        if existing:
            logger.info(
                f"Demo DB already has {len(existing)} papers; skipping duplicate seed."
            )
            return {"step": "search", "inserted": 0, "existing": len(existing)}
        session.add_all(sample_papers)
        session.commit()

    logger.info(f"Retrieved {len(sample_papers)} candidate papers (status=NOT_PROCESSED).")
    return {"step": "search", "inserted": len(sample_papers)}


def _demo_clean(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    with get_session_context(db_engine=engine) as session:
        papers = list(session.exec(select(PaperMetadata)).all())
        seen_dois = set()
        dedup_removed = 0
        empty_removed = 0
        for p in papers:
            if not (p.title and p.title.strip()) or not (p.abstract and p.abstract.strip()):
                session.delete(p)
                empty_removed += 1
                continue
            if p.doi in seen_dois:
                session.delete(p)
                dedup_removed += 1
            else:
                seen_dois.add(p.doi)
        session.commit()
    logger.info(
        f"Deduplicated {dedup_removed} duplicate DOI(s) and removed {empty_removed} empty-abstract paper(s)."
    )
    return {
        "step": "clean",
        "dedup_removed": dedup_removed,
        "empty_removed": empty_removed,
    }


def _demo_relevance(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    usable = 0
    discarded = 0
    with get_session_context(db_engine=engine) as session:
        papers = list(
            session.exec(
                select(PaperMetadata).where(
                    PaperMetadata.status == PaperStatus.NOT_PROCESSED
                )
            ).all()
        )
        for p in papers:
            is_comp_review = "review" in p.title.lower() or "simulation" in p.title.lower()
            is_rel = not is_comp_review
            conf = 0.14 if is_comp_review else 0.95
            explanation = (
                "Review / computational-only study without experimental self-assembly data."
                if is_comp_review
                else "Reports experimental peptide self-assembly conditions and characterization."
            )
            p.status = PaperStatus.USABLE if is_rel else PaperStatus.DISCARD
            session.add(p)
            session.add(
                PaperRelevance(
                    paper_id=p.id,
                    detected_clues=json.dumps(
                        ["molecular dynamics", "review"]
                        if is_comp_review
                        else ["self-assembly", "TEM", "pH"]
                    ),
                    explanation=explanation,
                    is_relevant=is_rel,
                    confidence=conf,
                    provider="demo",
                    model_name="demo-screener",
                    run_identifier=DEMO_RUN_ID,
                )
            )
            if is_rel:
                usable += 1
            else:
                discarded += 1
            logger.info(
                f"Paper {p.id} (PMID {p.pmid}): is_relevant={is_rel} (conf={conf:.2f}) -> {p.status.value}"
            )
        session.commit()
    return {"step": "relevance", "usable": usable, "discarded": discarded}


def _demo_fetch_fulltext(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    fetched = 0
    with get_session_context(db_engine=engine) as session:
        papers = list(
            session.exec(
                select(PaperMetadata).where(PaperMetadata.status == PaperStatus.USABLE)
            ).all()
        )
        for p in papers:
            existing = session.exec(
                select(PaperContent).where(PaperContent.paper_id == p.id)
            ).first()
            if existing:
                continue
            fmt = "xml" if p.pmcid else "pdf"
            session.add(
                PaperContent(
                    paper_id=p.id,
                    format=fmt,
                    source="demo_pmc" if fmt == "xml" else "demo_openalex",
                    blob_url=f"demo://papers/{fmt}/{p.pmid}.{fmt}",
                    run_identifier=DEMO_RUN_ID,
                )
            )
            fetched += 1
            logger.info(f"Fetched {fmt.upper()} fulltext for paper {p.id} (PMID {p.pmid}).")
        session.commit()
    return {"step": "fetch_fulltext", "fetched": fetched}


def _demo_extraction(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    templates = {
        "18052148": dict(
            sequence="FF",
            length=2,
            type="hydrogel",
            n_terminal="Fmoc",
            c_terminal="Free",
            temperature=25.0,
            ph=7.4,
            peptide_concentration=5.0,
            solution="PBS",
            solvent="DMSO/PBS",
            volume_ratio="5/95",
            chirality="l",
            detection_method="TEM; AFM; Rheology",
            preparation_protocol="Dissolved in DMSO and diluted into PBS 1x (5/95 v/v) at 25 °C.",
            evidence_span=json.dumps(
                {
                    "sequence": "[sec-1/p-1/s-1]",
                    "conditions": "[sec-2/p-2/s-3]",
                    "morphology": "[sec-3/fig-1/caption]",
                }
            ),
        ),
        "10468553": dict(
            sequence="KLVFF",
            length=5,
            type="fiber",
            n_terminal="Free",
            c_terminal="Free",
            temperature=37.0,
            ph=7.0,
            peptide_concentration=1.0,
            solution="Tris-HCl",
            solvent="H2O",
            volume_ratio="1",
            chirality="l",
            detection_method="ThT; CD; TEM",
            preparation_protocol="Incubated at 37 °C in pH 7.0 buffer for 24 h.",
            evidence_span=json.dumps(
                {
                    "sequence": "[sec-1/p-2/s-1]",
                    "conditions": "[sec-2/p-1/s-2]",
                    "morphology": "[sec-3/p-4/s-1]",
                }
            ),
        ),
        "23456789": dict(
            sequence="IIIK",
            length=4,
            type="tube",
            n_terminal="Acetylation",
            c_terminal="Amidation",
            temperature=25.0,
            ph=6.8,
            peptide_concentration=2.0,
            solution="H2O",
            solvent="H2O",
            volume_ratio="1",
            chirality="l",
            detection_method="Cryo-EM; AFM",
            preparation_protocol="Lyophilized Ac-IIIK-NH2 dissolved in Milli-Q water at 2.0 mg/ml.",
            evidence_span=json.dumps(
                {
                    "sequence": "[sec-1/p-1/s-2]",
                    "conditions": "[sec-2/table-1/row-2]",
                    "morphology": "[sec-3/fig-2/caption]",
                }
            ),
        ),
    }

    extracted = 0
    with get_session_context(db_engine=engine) as session:
        papers = list(
            session.exec(
                select(PaperMetadata).where(PaperMetadata.status == PaperStatus.USABLE)
            ).all()
        )
        for p in papers:
            tpl = templates.get(p.pmid or "")
            if not tpl:
                continue
            existing_llm = session.exec(
                select(PeptideEntry).where(
                    PeptideEntry.paper_id == p.id,
                    PeptideEntry.extraction_mode == ExtractionMode.LLM,
                )
            ).first()
            if existing_llm:
                continue

            # Add golden manual baseline for the golden_benchmark paper
            if p.dataset_subset == DatasetSubset.GOLDEN_BENCHMARK:
                existing_man = session.exec(
                    select(PeptideEntry).where(
                        PeptideEntry.paper_id == p.id,
                        PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
                    )
                ).first()
                if not existing_man:
                    session.add(
                        PeptideEntry(
                            paper_id=p.id,
                            pmid=p.pmid or "",
                            database="SAPDB; PeptideMiner",
                            extraction_mode=ExtractionMode.MANUAL,
                            run_identifier=0,
                            **{k: v for k, v in tpl.items() if k != "evidence_span"},
                        )
                    )

            session.add(
                PeptideEntry(
                    paper_id=p.id,
                    pmid=p.pmid or "",
                    extraction_mode=ExtractionMode.LLM,
                    model="demo-extractor",
                    run_identifier=DEMO_RUN_ID,
                    **tpl,
                )
            )
            extracted += 1
            logger.info(
                f"Extracted sequence={tpl['sequence']} ({tpl['type']}) from paper {p.id} with evidence_span."
            )
        session.commit()
    return {"step": "extraction", "extracted": extracted}


def _demo_evaluation(db_path: str) -> Dict[str, Any]:
    engine = get_engine(f"sqlite:///{Path(db_path).resolve()}")
    evaluated = 0
    with get_session_context(db_engine=engine) as session:
        entries = list(
            session.exec(
                select(PeptideEntry).where(
                    PeptideEntry.extraction_mode == ExtractionMode.LLM
                )
            ).all()
        )
        for e in entries:
            existing = session.exec(
                select(ExtractionEvaluation).where(
                    ExtractionEvaluation.peptide_entry_id == e.id
                )
            ).first()
            if existing:
                continue
            session.add(
                ExtractionEvaluation(
                    peptide_entry_id=e.id,
                    groundness_rate=0.96,
                    faithfulness_rate=0.94,
                    completeness_rate=0.91,
                    explanation="All sequence, condition, and morphology claims grounded in cited spans.",
                    run_identifier=DEMO_RUN_ID,
                )
            )
            evaluated += 1
            logger.info(
                f"Evaluated entry {e.id} ({e.sequence}): groundness=0.96, faithfulness=0.94, completeness=0.91"
            )
        session.commit()
    return {"step": "evaluation", "evaluated": evaluated}


def _run_full_demo(db_path: str) -> Dict[str, Any]:
    s1 = _demo_search(db_path)
    s2 = _demo_clean(db_path)
    s3 = _demo_relevance(db_path)
    s4 = _demo_fetch_fulltext(db_path)
    s5 = _demo_extraction(db_path)
    s6 = _demo_evaluation(db_path)
    return {"steps": [s1, s2, s3, s4, s5, s6]}


def submit_demo_step(step: str, db_path: str) -> Dict[str, Any]:
    """Submit a mocked demo pipeline step to the background JobManager."""
    dispatch = {
        "search": _demo_search,
        "clean": _demo_clean,
        "relevance": _demo_relevance,
        "fetch_fulltext": _demo_fetch_fulltext,
        "extraction": _demo_extraction,
        "evaluation": _demo_evaluation,
        "all": _run_full_demo,
    }
    if step not in dispatch:
        raise ValueError(f"Unknown demo step: {step}")
    job = job_manager.submit(f"demo:{step}", dispatch[step], db_path)
    return job.to_dict()
