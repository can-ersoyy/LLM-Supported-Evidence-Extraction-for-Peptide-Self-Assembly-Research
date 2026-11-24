"""Evidence-grounded disagreement auditor comparing LLM extractions against golden-dataset records.

Resolves LLM ``evidence_span`` tags back to verbatim source sentences or cropped table
images via ``parsers.py`` and invokes an independent Adjudicator LLM/VLM to distinguish:
- True LLM extraction errors (``golden_correct_llm_wrong`` / ``hallucinated_or_out_of_scope``)
- Suspected errors or omissions in the golden dataset (``llm_correct_golden_wrong`` / ``valid_uncurated_in_golden``)
- Surface-form synonyms needing normalizer aliases (``equivalent_normalization``)
- Ambiguous cases needing quick human review (``ambiguous_needs_human``)
"""

from __future__ import annotations

import base64
import csv
import json
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlmodel import Session, select

from peptide_extractor.db import (
    AuditType,
    DisagreementVerdict,
    ExtractionMode,
    GoldenDisagreementAudit,
    PeptideEntry,
    get_session_context,
    init_db,
)
from peptide_extractor.db.paper_content import get_paper_content_by_paper_id
from peptide_extractor.golden_benchmark import (
    _match_entries_for_sequence,
    _values_match_normalized,
)
from peptide_extractor.llm import LLMController
from peptide_extractor.parsers import (
    Node,
    TableNode,
    get_node_by_id,
    parse_document,
)
from peptide_extractor.utils import retry_on_external_error

logger = logging.getLogger(__name__)

DEFAULT_AUDIT_FIELDS: tuple[str, ...] = (
    "type",
    "n_terminal",
    "c_terminal",
    "temperature",
    "ph",
    "peptide_concentration",
    "solution",
    "volume_ratio",
    "incubation_period",
    "peptide_conjugate_mixture",
    "linear_cyclic",
    "detection_method",
)

FIELD_ADJUDICATOR_SYSTEM_PROMPT = """
You are an impartial scientific curation adjudicator auditing a disagreement between:
- A manually curated golden-dataset record (from SAPDB / PeptideMiner)
- An LLM-extracted record

You are given the field name, peptide sequence, both values, and the resolved source evidence (verbatim quote and/or table image) from the original paper.

Choose EXACTLY ONE verdict:
- `llm_correct_golden_wrong`: The paper evidence clearly supports the LLM value and contradicts the golden-dataset value (e.g., typo, wrong unit, or misread table cell in the curated DB).
- `golden_correct_llm_wrong`: The paper evidence clearly supports the golden-dataset value and the LLM extracted the wrong value.
- `equivalent_normalization`: Both values are scientifically equivalent ways of writing the same experimental condition (e.g., "EtOAc" vs "ethyl acetate", "H2O" vs "water", "RT" vs "25.0").
- `ambiguous_needs_human`: The provided evidence snippet does not contain enough information to conclusively determine which value is right.

Provide a confidence score in [0.0, 1.0] and a concise 1–2 sentence reason citing the exact evidence.
"""

UNMATCHED_ENTRY_ADJUDICATOR_SYSTEM_PROMPT = """
You are an impartial scientific curation adjudicator auditing an LLM-extracted peptide self-assembly entry that does NOT appear in the manually curated golden dataset (SAPDB / PeptideMiner).

You are given the LLM entry details and the resolved source evidence (verbatim quote and/or table image) from the original paper.

Choose EXACTLY ONE verdict:
- `valid_uncurated_in_golden`: The paper genuinely reports an experimental test of this peptide under these conditions, and the golden dataset simply omitted it.
- `hallucinated_or_out_of_scope`: The entry is hallucinated, misread, or refers only to cited prior literature rather than an experiment performed in this paper.
- `ambiguous_needs_human`: The provided evidence snippet does not conclusively confirm or refute the entry.

Provide a confidence score in [0.0, 1.0] and a concise 1–2 sentence reason citing the exact evidence.
"""


class FieldAdjudicationVerdict(BaseModel):
    """Structured adjudicator output for a matched-pair field mismatch."""

    verdict: Literal[
        "llm_correct_golden_wrong",
        "golden_correct_llm_wrong",
        "equivalent_normalization",
        "ambiguous_needs_human",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class UnmatchedEntryAdjudicationVerdict(BaseModel):
    """Structured adjudicator output for an unmatched LLM-only entry."""

    verdict: Literal[
        "valid_uncurated_in_golden",
        "hallucinated_or_out_of_scope",
        "ambiguous_needs_human",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


@dataclass
class DisagreementCandidate:
    """An unadjudicated disagreement between a golden record and an LLM record."""

    pmid: str
    paper_id: int | None
    sequence: str
    audit_type: AuditType
    field_name: str
    manual_entry: PeptideEntry | None
    llm_entry: PeptideEntry
    manual_value: str | None
    llm_value: str | None


def _parse_evidence_span(raw_span: str | None) -> dict[str, str]:
    if not raw_span:
        return {}
    try:
        parsed = json.loads(raw_span)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items() if v}
    except (ValueError, TypeError):
        pass
    return {}


def resolve_evidence_for_field(
    paper_tree: Node | None,
    evidence_span_json: str | None,
    field_name: str,
) -> tuple[str | None, str, bytes | None]:
    """Resolve an entry's evidence_span tag to (tag, text_quote, optional_image_bytes)."""
    span_map = _parse_evidence_span(evidence_span_json)
    tag = span_map.get(field_name) or span_map.get("sequence") or span_map.get("type")
    if not tag or paper_tree is None:
        return tag, "", None

    node = get_node_by_id(paper_tree, tag)
    if node is None:
        return tag, "", None

    image_bytes = getattr(node, "image_bytes", None)
    if isinstance(node, TableNode):
        parts = []
        if node.caption:
            parts.append(f"Caption: {node.caption}")
        if node.columns:
            parts.append("Columns: " + " | ".join(node.columns))
        if node.rows:
            row_preview = "\n".join(" | ".join(r) for r in node.rows[:15])
            parts.append(row_preview)
        if node.footnotes:
            parts.append("Footnotes: " + " ".join(node.footnotes))
        return tag, "\n".join(parts).strip(), image_bytes

    quote_text = getattr(node, "content", None) or getattr(node, "caption", None) or ""
    return tag, quote_text.strip(), image_bytes


def collect_disagreements(
    manual_entries: Sequence[PeptideEntry],
    llm_entries: Sequence[PeptideEntry],
    *,
    fields: Sequence[str] = DEFAULT_AUDIT_FIELDS,
    include_unmatched_llm: bool = True,
    max_field_mismatches_per_pmid: int | None = 25,
    max_unmatched_per_pmid: int | None = 10,
    min_similarity: float = 0.1,
) -> list[DisagreementCandidate]:
    """Collect field-level mismatches on matched pairs plus unmatched LLM-only entries."""
    m_by_ps: dict[tuple[str, str], list[PeptideEntry]] = defaultdict(list)
    for e in manual_entries:
        m_by_ps[(e.pmid, e.sequence.upper())].append(e)

    l_by_ps: dict[tuple[str, str], list[PeptideEntry]] = defaultdict(list)
    for e in llm_entries:
        l_by_ps[(e.pmid, e.sequence.upper())].append(e)

    field_counts: dict[str, int] = defaultdict(int)
    unmatched_counts: dict[str, int] = defaultdict(int)
    candidates: list[DisagreementCandidate] = []

    for key in sorted(set(m_by_ps.keys()) | set(l_by_ps.keys())):
        pmid, seq = key
        pairs, _, unmatched_l = _match_entries_for_sequence(
            m_by_ps.get(key, []),
            l_by_ps.get(key, []),
            min_similarity=min_similarity,
        )

        for m_entry, l_entry in pairs:
            for fname in fields:
                if (
                    max_field_mismatches_per_pmid is not None
                    and field_counts[pmid] >= max_field_mismatches_per_pmid
                ):
                    break
                mv = getattr(m_entry, fname, None)
                lv = getattr(l_entry, fname, None)
                m_ok = mv is not None and str(mv).strip() != ""
                l_ok = lv is not None and str(lv).strip() != ""
                if not (m_ok and l_ok):
                    continue
                _, norm_match = _values_match_normalized(fname, m_entry, l_entry)
                if not norm_match:
                    candidates.append(
                        DisagreementCandidate(
                            pmid=pmid,
                            paper_id=l_entry.paper_id or m_entry.paper_id,
                            sequence=seq,
                            audit_type=AuditType.FIELD_MISMATCH,
                            field_name=fname,
                            manual_entry=m_entry,
                            llm_entry=l_entry,
                            manual_value=str(mv),
                            llm_value=str(lv),
                        )
                    )
                    field_counts[pmid] += 1

        if include_unmatched_llm:
            for l_entry in unmatched_l:
                if (
                    max_unmatched_per_pmid is not None
                    and unmatched_counts[pmid] >= max_unmatched_per_pmid
                ):
                    break
                summary_val = (
                    f"seq={l_entry.sequence}, type={l_entry.type}, "
                    f"N={l_entry.n_terminal}, C={l_entry.c_terminal}, "
                    f"solution={l_entry.solution or l_entry.solvent}, "
                    f"T={l_entry.temperature}, pH={l_entry.ph}, "
                    f"conc={l_entry.peptide_concentration}"
                )
                candidates.append(
                    DisagreementCandidate(
                        pmid=pmid,
                        paper_id=l_entry.paper_id,
                        sequence=seq,
                        audit_type=AuditType.LLM_ONLY_ENTRY,
                        field_name="entry",
                        manual_entry=None,
                        llm_entry=l_entry,
                        manual_value=None,
                        llm_value=summary_val,
                    )
                )
                unmatched_counts[pmid] += 1

    return candidates


def _build_adjudicator_payload(prompt_text: str, image_bytes: bytes | None) -> str | list[dict[str, Any]]:
    if image_bytes:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        return [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    return prompt_text


def adjudicate_candidate(
    candidate: DisagreementCandidate,
    *,
    evidence_tag: str | None,
    source_quote: str,
    image_bytes: bytes | None,
    field_adjudicator: LLMController,
    unmatched_adjudicator: LLMController,
    max_retries: int = 0,
    retry_delay: float = 2.0,
) -> tuple[DisagreementVerdict, float | None, str]:
    """Run the independent Adjudicator LLM/VLM on a single disagreement candidate."""
    if not source_quote and not image_bytes:
        return (
            DisagreementVerdict.AMBIGUOUS_NEEDS_HUMAN,
            0.0,
            f"Could not resolve evidence tag '{evidence_tag}' in source document.",
        )

    if candidate.audit_type == AuditType.FIELD_MISMATCH:
        prompt_text = (
            f"PMID: {candidate.pmid}\n"
            f"Peptide Sequence: {candidate.sequence}\n"
            f"Field: {candidate.field_name}\n"
            f"Golden Dataset (Manual) Value: {candidate.manual_value}\n"
            f"LLM Extracted Value: {candidate.llm_value}\n"
            f"Cited Evidence Tag: {evidence_tag or 'None'}\n"
            f"Resolved Source Evidence:\n{source_quote or '(see attached table/figure image)'}"
        )
        payload = _build_adjudicator_payload(prompt_text, image_bytes)
        res: FieldAdjudicationVerdict = retry_on_external_error(
            field_adjudicator.extract_peptides,
            payload,
            stream=False,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context=f"audit field {candidate.pmid}/{candidate.sequence}.{candidate.field_name}",
        )
        return DisagreementVerdict(res.verdict), res.confidence, res.reason

    prompt_text = (
        f"PMID: {candidate.pmid}\n"
        f"Unmatched LLM Entry: {candidate.llm_value}\n"
        f"Cited Evidence Tag: {evidence_tag or 'None'}\n"
        f"Resolved Source Evidence:\n{source_quote or '(see attached table/figure image)'}"
    )
    payload = _build_adjudicator_payload(prompt_text, image_bytes)
    res_u: UnmatchedEntryAdjudicationVerdict = retry_on_external_error(
        unmatched_adjudicator.extract_peptides,
        payload,
        stream=False,
        max_retries=max_retries,
        retry_delay=retry_delay,
        context=f"audit unmatched {candidate.pmid}/{candidate.sequence}",
    )
    return DisagreementVerdict(res_u.verdict), res_u.confidence, res_u.reason


def format_audit_report(audits: Sequence[GoldenDisagreementAudit]) -> str:
    """Format audited disagreements into a concise, human-reviewable Markdown report."""
    counts = Counter(a.verdict.value if hasattr(a.verdict, "value") else str(a.verdict) for a in audits)
    lines = [
        f"# Golden Dataset Disagreement Audit ({len(audits)} items)",
        "",
        "## Verdict Summary",
        "",
        "| Verdict | Count |",
        "| :--- | ---: |",
    ]
    for verdict in DisagreementVerdict:
        lines.append(f"| `{verdict.value}` | {counts.get(verdict.value, 0)} |")

    grouped: dict[str, list[GoldenDisagreementAudit]] = defaultdict(list)
    for a in audits:
        v_key = a.verdict.value if hasattr(a.verdict, "value") else str(a.verdict)
        grouped[v_key].append(a)

    for verdict in DisagreementVerdict:
        rows = grouped.get(verdict.value, [])
        if not rows:
            continue
        lines.extend(
            [
                "",
                f"## `{verdict.value}` ({len(rows)})",
                "",
                "| PMID | Seq | Field | Manual | LLM | Tag | Conf | Reason |",
                "| :--- | :--- | :--- | :--- | :--- | :--- | ---: | :--- |",
            ]
        )
        for r in rows:
            conf_str = f"{r.confidence:.2f}" if r.confidence is not None else "-"
            reason_clean = (r.reason or "").replace("\n", " ").replace("|", "/")
            man_clean = (r.manual_value or "-").replace("\n", " ").replace("|", "/")
            llm_clean = (r.llm_value or "-").replace("\n", " ").replace("|", "/")
            lines.append(
                f"| {r.pmid} | `{r.sequence}` | `{r.field_name}` | {man_clean} | {llm_clean} | "
                f"`{r.evidence_tag or '-'}` | {conf_str} | {reason_clean} |"
            )

    return "\n".join(lines)


def export_audit_csv(audits: Sequence[GoldenDisagreementAudit], output_path: str | Path) -> Path:
    """Export audited disagreements to a CSV file for spreadsheet review."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "pmid",
        "paper_id",
        "manual_entry_id",
        "llm_entry_id",
        "sequence",
        "audit_type",
        "field_name",
        "manual_value",
        "llm_value",
        "evidence_tag",
        "source_quote",
        "verdict",
        "confidence",
        "reason",
        "adjudicator_provider",
        "adjudicator_model",
        "human_verdict",
        "human_notes",
        "llm_run_identifier",
        "audit_run_identifier",
    ]
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for a in audits:
            writer.writerow(
                {
                    "id": a.id,
                    "pmid": a.pmid,
                    "paper_id": a.paper_id,
                    "manual_entry_id": a.manual_entry_id,
                    "llm_entry_id": a.llm_entry_id,
                    "sequence": a.sequence,
                    "audit_type": a.audit_type.value if hasattr(a.audit_type, "value") else a.audit_type,
                    "field_name": a.field_name,
                    "manual_value": a.manual_value,
                    "llm_value": a.llm_value,
                    "evidence_tag": a.evidence_tag,
                    "source_quote": a.source_quote,
                    "verdict": a.verdict.value if hasattr(a.verdict, "value") else a.verdict,
                    "confidence": a.confidence,
                    "reason": a.reason,
                    "adjudicator_provider": a.adjudicator_provider,
                    "adjudicator_model": a.adjudicator_model,
                    "human_verdict": (
                        a.human_verdict.value
                        if a.human_verdict is not None and hasattr(a.human_verdict, "value")
                        else a.human_verdict
                    ),
                    "human_notes": a.human_notes,
                    "llm_run_identifier": a.llm_run_identifier,
                    "audit_run_identifier": a.audit_run_identifier,
                }
            )
    return target


def audit_run_against_golden(
    llm_run_identifier: int,
    *,
    manual_run_identifier: int = 0,
    pmids: Sequence[str] | None = None,
    provider: str = "google_genai",
    model_name: str = "gemini-3.5-flash-lite",
    base_url: str | None = None,
    fields: Sequence[str] = DEFAULT_AUDIT_FIELDS,
    include_unmatched_llm: bool = True,
    max_field_mismatches_per_pmid: int | None = 25,
    max_unmatched_per_pmid: int | None = 10,
    min_similarity: float = 0.1,
    save: bool = True,
    audit_run_identifier: int | None = None,
    export_markdown_path: str | Path | None = None,
    export_csv_path: str | Path | None = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
    session: Session | None = None,
) -> tuple[list[GoldenDisagreementAudit], str]:
    """Audit disagreements between an LLM run and golden manual entries against source papers.

    Resolves each entry's ``evidence_span`` tag via ``parsers.parse_document`` +
    ``get_node_by_id``, runs an independent Adjudicator LLM/VLM, optionally persists
    ``GoldenDisagreementAudit`` rows to the database, and optionally exports Markdown/CSV reports.
    """
    resolved_audit_run_id = audit_run_identifier or int(time.time())

    field_adjudicator = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=FieldAdjudicationVerdict,
        system_prompt=FIELD_ADJUDICATOR_SYSTEM_PROMPT,
        base_url=base_url,
    )
    unmatched_adjudicator = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=UnmatchedEntryAdjudicationVerdict,
        system_prompt=UNMATCHED_ENTRY_ADJUDICATOR_SYSTEM_PROMPT,
        base_url=base_url,
    )

    def _run(s: Session) -> tuple[list[GoldenDisagreementAudit], str]:
        m_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == manual_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.MANUAL,
        )
        l_stmt = select(PeptideEntry).where(
            PeptideEntry.run_identifier == llm_run_identifier,
            PeptideEntry.extraction_mode == ExtractionMode.LLM,
        )
        if pmids is not None:
            pmid_list = list(pmids)
            m_stmt = m_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]
            l_stmt = l_stmt.where(PeptideEntry.pmid.in_(pmid_list))  # type: ignore[union-attr]

        manual_entries = s.exec(m_stmt).all()
        llm_entries = s.exec(l_stmt).all()

        candidates = collect_disagreements(
            manual_entries,
            llm_entries,
            fields=fields,
            include_unmatched_llm=include_unmatched_llm,
            max_field_mismatches_per_pmid=max_field_mismatches_per_pmid,
            max_unmatched_per_pmid=max_unmatched_per_pmid,
            min_similarity=min_similarity,
        )

        # Parse each paper's document tree once (cached by paper_id)
        paper_trees: dict[int, Node | None] = {}
        audits: list[GoldenDisagreementAudit] = []

        for cand in candidates:
            pid = cand.paper_id
            if pid is not None and pid not in paper_trees:
                paper_content = get_paper_content_by_paper_id(pid, s)
                if paper_content is None:
                    paper_trees[pid] = None
                else:
                    try:
                        raw = paper_content.get_content()
                        paper_trees[pid] = parse_document(
                            raw,
                            format=paper_content.format or "auto",
                            crop_pdf_tables=(paper_content.format == "pdf"),
                            crop_pdf_images=(paper_content.format == "pdf"),
                        )
                    except (ValueError, RuntimeError, OSError) as e:
                        logger.warning(f"Could not parse PaperContent for paper {pid}: {e}")
                        paper_trees[pid] = None

            tree = paper_trees.get(pid) if pid is not None else None
            tag, quote, img_bytes = resolve_evidence_for_field(
                tree, cand.llm_entry.evidence_span, cand.field_name
            )
            verdict, confidence, reason = adjudicate_candidate(
                cand,
                evidence_tag=tag,
                source_quote=quote,
                image_bytes=img_bytes,
                field_adjudicator=field_adjudicator,
                unmatched_adjudicator=unmatched_adjudicator,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )

            audit_row = GoldenDisagreementAudit(
                pmid=cand.pmid,
                paper_id=cand.paper_id,
                manual_entry_id=cand.manual_entry.id if cand.manual_entry else None,
                llm_entry_id=cand.llm_entry.id,
                sequence=cand.sequence,
                audit_type=cand.audit_type,
                field_name=cand.field_name,
                manual_value=cand.manual_value,
                llm_value=cand.llm_value,
                evidence_tag=tag,
                source_quote=quote or None,
                verdict=verdict,
                confidence=confidence,
                reason=reason,
                adjudicator_provider=provider,
                adjudicator_model=model_name,
                llm_run_identifier=llm_run_identifier,
                audit_run_identifier=resolved_audit_run_id,
            )
            audits.append(audit_row)

        if save and audits:
            s.add_all(audits)
            s.commit()
            for row in audits:
                s.refresh(row)

        report_md = format_audit_report(audits)
        if export_markdown_path is not None:
            md_target = Path(export_markdown_path)
            md_target.parent.mkdir(parents=True, exist_ok=True)
            md_target.write_text(report_md, encoding="utf-8")
        if export_csv_path is not None:
            export_audit_csv(audits, export_csv_path)

        return audits, report_md

    if session is not None:
        return _run(session)
    init_db()
    with get_session_context() as s:
        return _run(s)
