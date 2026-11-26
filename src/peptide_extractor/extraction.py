from peptide_extractor.logging_utils import logger
import re
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from peptide_extractor.db.peptide_entry import (
    EvidenceSpan,
    PeptideAssemblyType,
    PeptideConjugateMixture,
    PeptideExtractionResult,
    PeptideLLMEntry,
    PeptideLinearity,
)
from peptide_extractor.llm import LLMController
from peptide_extractor.utils import retry_on_external_error

LOCATION_GROUNDING_RULES = """
LOCATION GROUNDING RULES:
1. Pure Location Annotations Only:
   Whenever you extract ANY non-null property in an entry (e.g., sequence, type, temperature, ph, peptide_concentration, solution, incubation_period), you MUST provide ONLY the location annotation tag in the corresponding field inside `evidence_span` where this information is found.
2. Format:
   Locations in the text are tagged in square brackets, e.g., `[sec-1/p-2/s-1]`, `[sec-5/p-3/sec-1/p-1/s-11]`, `[sec-1/table-0]`, `[sec-2/fig-0]`. Provide ONLY the exact bracketed location tag string, e.g., "[sec-1/p-2/s-1]".
   When a table or figure is attached as an image, cite its bracketed location tag (e.g., "[sec-1/table-0]" or "[sec-2/fig-0]").
3. DO NOT include any verbatim text quotes, excerpts, or explanations inside `evidence_span`. Strictly output only the location identifier.
4. There can be at most 1 location tag provided per field. Different fields may share the same location tag if they appear in the same sentence or paragraph.
5. If a property was NOT experimentally determined or mentioned in the text, leave both the property and its corresponding `evidence_span` field as null.
"""

EXTRACTION_SYSTEM_PROMPT_V3 = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema. 
There can be several entries in a single paper, be careful!

{LOCATION_GROUNDING_RULES}
"""

EXTRACTION_SYSTEM_PROMPT_V5 = """
You will receive the full text of a scientific paper and your goal is to extract experimental peptide self-assembly information according to the given schema.
In the paper, there can be multiple entries, possibly with pretty similar conditions across different columns of the extraction schema.
For example, if you see that for a given peptide sequence, two different temperature conditions are tested, you need to have an extraction for each of these temperature values.
This applies to other fields as well (e.g. pH, incubation_period, solution, solvent etc.)

{LOCATION_GROUNDING_RULES}
"""


PDF_EXTRACTION_SYSTEM_PROMPT = """
You are an expert scientific curator extracting experimental peptide self-assembly data from an attached PDF paper into the given JSON schema.

EXHAUSTIVE EXTRACTION RULES (CRITICAL):
1. Do NOT summarize, sample, or stop after a few examples. You MUST extract EVERY experimental peptide condition reported anywhere in the paper (prose, tables, table footnotes, figures, and captions).
2. Matrix / Screening Tables: Papers often contain tables testing N peptides across M solvents, concentrations, pH values, or temperatures. You MUST unroll the entire matrix and emit a separate entry for EVERY SINGLE TESTED `(peptide, condition)` cell—including negative / non-assembling results (e.g., soluble, precipitate, no gel -> `type: "none"`). Never skip non-gelling or negative rows/columns.
3. Cross-Reference Shared Conditions: Tables frequently omit experimental parameters that are stated in the Methods/Results text or table footnotes (such as temperature, incubation period, pH, N-/C-terminal protecting groups, preparation protocol, or detection methods). Apply those shared conditions to every relevant table entry.
4. Sequences & Terminals:
   - Convert 3-letter amino acid sequences (e.g., Boc-Ser-Val-Gly-OCH2Ph) into uppercase 1-letter codes (`sequence: "SVG"`).
   - Put N-terminal and C-terminal caps/protecting groups in `n_terminal` and `c_terminal` rather than inside `sequence`.
5. Multiple Conditions = Multiple Entries: If the same peptide is tested in multiple solvents, temperatures, pH values, or concentrations, emit a separate entry for each tested combination.
"""

EXTRACTION_SYSTEM_PROMPT_V6 = """
You are an expert scientific curator extracting experimental peptide self-assembly data from given papers.

EXHAUSTIVE EXTRACTION RULES (CRITICAL):
1. Do NOT summarize, sample, or stop after a few examples. You MUST extract EVERY experimental peptide condition reported anywhere in the paper (prose, tables, table footnotes, figures, and captions).
2. Matrix / Screening Tables: Papers often contain tables testing N peptides across M solvents, concentrations, pH values, or temperatures. You MUST unroll the entire matrix and emit a separate entry for EVERY SINGLE TESTED `(peptide, condition)` cell—including negative / non-assembling results (e.g., soluble, precipitate, no gel -> `type: "none"`). Never skip non-gelling or negative rows/columns.
3. Cross-Reference Shared Conditions: Tables frequently omit experimental parameters that are stated in the Methods/Results text or table footnotes (such as temperature, incubation period, pH, N-/C-terminal protecting groups, preparation protocol, or detection methods). Apply those shared conditions to every relevant table entry.
4. Sequences & Terminals:
   - Convert 3-letter amino acid sequences (e.g., Boc-Ser-Val-Gly-OCH2Ph) into uppercase 1-letter codes (`sequence: "SVG"`).
   - Put N-terminal and C-terminal caps/protecting groups in `n_terminal` and `c_terminal` rather than inside `sequence`.
5. Multiple Conditions = Multiple Entries: If the same peptide is tested in multiple solvents, temperatures, pH values, or concentrations, emit a separate entry for each tested combination.

{LOCATION_GROUNDING_RULES}
"""

SHARED_CONDITIONS_SYSTEM_PROMPT = f"""
You are an expert scientific curator. Read the location-annotated paper text and extract TWO things:
1. `shared_conditions`: Paper-wide experimental defaults that apply across the screening/assembly experiments (such as `temperature` in °C, `incubation_period` in minutes, `ph`, `n_terminal`, `c_terminal`, `peptide_concentration` in mg/ml, `solution`, `solvent`, `volume_ratio`, `peptide_conjugate_mixture`, `linear_cyclic`, `chirality`, `detection_method`, `preparation_protocol`), along with their exact bracketed `evidence_span` location tags (e.g. `"[sec-0/p-2/s-4]"`).
2. `prose_entries`: Only experimental peptide self-assembly entries reported in prose that are NOT already covered by the paper's tables (tables will be extracted separately from their images). If all entries are in tables, return `prose_entries: []`.

Be thorough on `shared_conditions`—convert hours to minutes for `incubation_period` (e.g. 24 h -> 1440) and convert RT/room temperature to 25.0 °C.

{LOCATION_GROUNDING_RULES}
"""

TABLE_EXTRACTION_SYSTEM_PROMPT = """
You are an expert scientific curator extracting experimental peptide self-assembly data from a single table image into a compact row-grouped schema.

CRITICAL RULES:
1. Extract EVERY SINGLE ROW and EVERY SINGLE TESTED CELL in the attached table image. Never summarize, sample, or stop after the first row.
2. Group by table row (`CompactTableRow`):
   - `sequence`: Uppercase 1-letter IUPAC amino acid code (convert 3-letter codes like `Ser-Val-Gly` or `S-V-G` to `SVG`).
   - `n_terminal` / `c_terminal`: Only set if row-specific or different from the paper-wide shared defaults.
   - `conditions`: Emit one `CompactTableCondition` for EVERY tested column/condition in that row—including negative/non-assembling cells (e.g., insoluble, homogeneous solution, precipitate -> `type: "none"`). For gels in organic solvents use `type: "organogel"`, for aqueous gels use `type: "hydrogel"`.
3. Do NOT repeat paper-wide shared conditions (like shared `temperature`, `incubation_period`, `preparation_protocol`, `detection_method`) in every cell unless the table or a specific cell overrides them. They will be merged automatically.
"""


class SharedExperimentalConditions(BaseModel):
    """Paper-wide or table-wide experimental defaults merged into extracted entries."""

    n_terminal: Optional[str] = None
    c_terminal: Optional[str] = None
    temperature: Optional[float] = Field(
        default=None,
        description="Reaction/assembly temperature in Celsius (convert RT to 25.0).",
    )
    ph: Optional[float] = None
    peptide_concentration: Optional[float] = Field(
        default=None,
        description="Concentration numeric value (prefer mg/ml; if mM/uM/M/wt%, set peptide_concentration_unit).",
    )
    peptide_concentration_unit: Optional[str] = Field(
        default=None,
        description="Unit for peptide_concentration (e.g. 'mg/ml', 'wt%', 'mM', 'uM', 'M').",
    )
    solution: Optional[str] = None
    solvent: Optional[str] = None
    volume_ratio: Optional[list[int]] = None
    incubation_period: Optional[int] = Field(
        default=None,
        description="Incubation duration in minutes (e.g. 24 h -> 1440).",
    )
    peptide_conjugate_mixture: Optional[PeptideConjugateMixture] = None
    conjugate_partner: Optional[str] = None
    linear_cyclic: Optional[PeptideLinearity] = None
    chirality: Optional[str] = None
    detection_method: Optional[list[str]] = None
    preparation_protocol: Optional[str] = None
    evidence_span: Optional[EvidenceSpan] = None


class PaperWideExtractionResult(BaseModel):
    """Pass 1 output: paper-wide shared conditions plus any prose-only entries."""

    shared_conditions: SharedExperimentalConditions = Field(
        default_factory=SharedExperimentalConditions
    )
    prose_entries: list[PeptideLLMEntry] = Field(default_factory=list)


class CompactTableCondition(BaseModel):
    """Per-cell condition within a table row (omits repeated paper-wide fields)."""

    solution: Optional[str] = None
    solvent: Optional[str] = None
    volume_ratio: Optional[list[int]] = None
    type: Optional[PeptideAssemblyType] = None
    peptide_concentration: Optional[float] = None
    peptide_concentration_unit: Optional[str] = None
    temperature: Optional[float] = None
    ph: Optional[float] = None
    incubation_period: Optional[int] = None


class CompactTableRow(BaseModel):
    """One peptide row in a screening/experimental table."""

    sequence: str = Field(
        description="Uppercase 1-letter IUPAC amino acid sequence (no spaces or hyphens)."
    )
    n_terminal: Optional[str] = None
    c_terminal: Optional[str] = None
    conditions: list[CompactTableCondition] = Field(default_factory=list)


class TableExtractionResult(BaseModel):
    """Pass 2 output: compact row-grouped extraction for a single table image."""

    table_defaults: Optional[SharedExperimentalConditions] = None
    rows: list[CompactTableRow] = Field(default_factory=list)


_TABLE_ATTACHED_RE = re.compile(
    r">>>\s*\[(TABLE IMAGE ATTACHED|SEE APPENDIX TABLE IMAGE|FIGURE IMAGE ATTACHED|SEE APPENDIX FIGURE IMAGE):\s*([^\]]+)\]\s*<<<"
)
_APPENDIX_HEADER_RE = re.compile(
    r"===\s*APPENDIX (TABLE|FIGURE) IMAGE:\s*\[([^\]]+)\]\s*==="
)
_EVIDENCE_FIELDS = set(EvidenceSpan.model_fields.keys())


def _is_figure_kind(kind_label: Optional[str], visual_id: Optional[str]) -> bool:
    if kind_label and "FIGURE" in kind_label.upper():
        return True
    if visual_id and ("fig-" in visual_id.lower() or visual_id.lower().startswith("fig")):
        return True
    return False


def _split_blocks_for_visual_extraction(
    blocks: list[dict[str, Any]],
) -> tuple[
    str,
    list[tuple[str, str, dict[str, Any]]],
    list[tuple[str, str, dict[str, Any]]],
]:
    """Split multimodal blocks into (prose_text, table_items, figure_items).

    Each visual item is a tuple of `(location_id, caption_text, image_block)`.
    """
    prose_parts: list[str] = []
    table_items: list[tuple[str, str, dict[str, Any]]] = []
    figure_items: list[tuple[str, str, dict[str, Any]]] = []
    # Maps location_id -> (kind, caption_text) where kind is "table" or "figure"
    pending_meta: dict[str, tuple[str, str]] = {}

    for idx, block in enumerate(blocks):
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            appendix_match = _APPENDIX_HEADER_RE.search(text)
            if appendix_match:
                kind_label = appendix_match.group(1).strip()
                vid = appendix_match.group(2).strip()
                kind = "figure" if _is_figure_kind(kind_label, vid) else "table"
                pending_meta[vid] = (kind, text.strip())
                continue

            prose_parts.append(text)
            for m in _TABLE_ATTACHED_RE.finditer(text):
                kind_label = m.group(1).strip()
                vid = m.group(2).strip()
                kind = "figure" if _is_figure_kind(kind_label, vid) else "table"
                # Capture the caption/footnotes lines right above the marker
                lines = text[: m.start()].splitlines()
                caption_lines = [
                    ln.strip() for ln in lines[-4:] if vid in ln or "Footnotes:" in ln
                ]
                caption = "\n".join(caption_lines) if caption_lines else f"[{vid}]"
                pending_meta[vid] = (kind, caption)
        elif btype == "image_url":
            vid: Optional[str] = None
            kind = "table"
            caption = ""
            if idx > 0 and blocks[idx - 1].get("type") == "text":
                prev_text = blocks[idx - 1].get("text", "")
                m_app = _APPENDIX_HEADER_RE.search(prev_text)
                m_att = _TABLE_ATTACHED_RE.search(prev_text)
                if m_app:
                    vid = m_app.group(2).strip()
                    kind = "figure" if _is_figure_kind(m_app.group(1), vid) else "table"
                elif m_att:
                    vid = m_att.group(2).strip()
                    kind = "figure" if _is_figure_kind(m_att.group(1), vid) else "table"
            if vid is None and pending_meta:
                vid = next(iter(pending_meta))
            if vid is not None and vid in pending_meta:
                kind, caption = pending_meta.pop(vid)
            elif vid is not None:
                caption = f"[{vid}]"
            else:
                vid = f"table-{len(table_items)}"
                caption = f"[{vid}]"

            if kind == "figure":
                figure_items.append((vid, caption, block))
            else:
                table_items.append((vid, caption, block))

    return "\n".join(prose_parts), table_items, figure_items


def _split_blocks_for_factored_extraction(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[tuple[str, str, dict[str, Any]]]]:
    """Backwards-compatible wrapper returning (prose_text, all_visual_items)."""
    prose_text, table_items, figure_items = _split_blocks_for_visual_extraction(blocks)
    return prose_text, table_items + figure_items


def _format_shared_conditions_summary(shared: SharedExperimentalConditions) -> str:
    """Format non-null shared conditions as concise context for Pass 2 table calls."""
    data = shared.model_dump(exclude={"evidence_span"}, exclude_none=True)
    if not data:
        return "None specified."
    return "\n".join(f"- {k}: {v}" for k, v in data.items())


def merge_table_result_with_shared(
    table_result: TableExtractionResult,
    shared: SharedExperimentalConditions,
    table_id: str,
) -> list[PeptideLLMEntry]:
    """Unroll a compact TableExtractionResult into full PeptideLLMEntry rows with merged defaults."""
    table_tag = table_id if table_id.startswith("[") else f"[{table_id}]"
    table_defaults = table_result.table_defaults or SharedExperimentalConditions()
    shared_ev = shared.evidence_span or EvidenceSpan()
    table_ev = table_defaults.evidence_span or EvidenceSpan()

    entries: list[PeptideLLMEntry] = []
    for row in table_result.rows:
        row_conditions = row.conditions or [CompactTableCondition()]
        for cell in row_conditions:
            ev_data: dict[str, Optional[str]] = {"sequence": table_tag}

            def _resolve(field: str, cell_val: Any = None, row_val: Any = None) -> Any:
                if cell_val is not None:
                    if field in _EVIDENCE_FIELDS:
                        ev_data[field] = table_tag
                    return cell_val
                if row_val is not None:
                    if field in _EVIDENCE_FIELDS:
                        ev_data[field] = table_tag
                    return row_val
                t_val = getattr(table_defaults, field, None)
                if t_val is not None:
                    if field in _EVIDENCE_FIELDS:
                        ev_data[field] = getattr(table_ev, field, None) or table_tag
                    return t_val
                s_val = getattr(shared, field, None)
                if s_val is not None and field in _EVIDENCE_FIELDS:
                    ev_data[field] = getattr(shared_ev, field, None)
                return s_val

            entry = PeptideLLMEntry(
                sequence=row.sequence,
                type=cell.type,
                n_terminal=_resolve("n_terminal", row_val=row.n_terminal),
                c_terminal=_resolve("c_terminal", row_val=row.c_terminal),
                temperature=_resolve("temperature", cell_val=cell.temperature),
                ph=_resolve("ph", cell_val=cell.ph),
                peptide_concentration=_resolve(
                    "peptide_concentration", cell_val=cell.peptide_concentration
                ),
                peptide_concentration_unit=_resolve(
                    "peptide_concentration_unit",
                    cell_val=cell.peptide_concentration_unit,
                ),
                solution=_resolve("solution", cell_val=cell.solution),
                solvent=(
                    _resolve("solvent", cell_val=cell.solvent)
                    if cell.solution is None or cell.solvent is not None
                    else None
                ),
                volume_ratio=_resolve("volume_ratio", cell_val=cell.volume_ratio),
                incubation_period=_resolve(
                    "incubation_period", cell_val=cell.incubation_period
                ),
                peptide_conjugate_mixture=_resolve("peptide_conjugate_mixture"),
                conjugate_partner=_resolve("conjugate_partner"),
                linear_cyclic=_resolve("linear_cyclic"),
                chirality=_resolve("chirality"),
                detection_method=_resolve("detection_method"),
                preparation_protocol=_resolve("preparation_protocol"),
                evidence_span=EvidenceSpan(**ev_data),
            )
            entries.append(entry)

    return entries


def run_factored_extraction(
    paper_content: Union[str, bytes, Path, list[dict[str, Any]]],
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
    filename: str = "paper.pdf",
    table_provider: Optional[str] = None,
    table_model_name: Optional[str] = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
) -> PeptideExtractionResult:
    """Two-pass extraction: Pass 1 extracts shared paper-wide conditions; Pass 2 extracts each table."""
    if not isinstance(paper_content, list):
        return run_extraction(
            paper_content,
            provider=provider,
            model_name=model_name,
            filename=filename,
            factored=False,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    prose_text, table_items, figure_items = _split_blocks_for_visual_extraction(
        paper_content
    )
    if not table_items and not figure_items:
        return run_extraction(
            paper_content,
            provider=provider,
            model_name=model_name,
            filename=filename,
            factored=False,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    # Pass 1: Extract paper-wide experimental defaults + prose-only entries
    shared_controller = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=PaperWideExtractionResult,
        system_prompt=SHARED_CONDITIONS_SYSTEM_PROMPT,
    )
    pass1_res: PaperWideExtractionResult = retry_on_external_error(
        shared_controller.extract_peptides,
        prose_text,
        stream=True,
        filename=filename,
        max_retries=max_retries,
        retry_delay=retry_delay,
        context=f"extraction pass1 {filename}",
    )
    shared = pass1_res.shared_conditions
    shared_summary = _format_shared_conditions_summary(shared)

    all_entries: list[PeptideLLMEntry] = list(pass1_res.prose_entries)

    # Pass 2: Extract each table image independently with compact row-grouped schema
    table_controller = LLMController(
        provider=table_provider or provider,
        model_name=table_model_name or model_name,
        structured_output=TableExtractionResult,
        system_prompt=TABLE_EXTRACTION_SYSTEM_PROMPT,
    )
    for table_id, caption_text, image_block in table_items:
        table_prompt_blocks = [
            {
                "type": "text",
                "text": (
                    f"Paper-wide shared conditions (merged automatically; only override if this table differs):\n"
                    f"{shared_summary}\n\n"
                    f"Table location ID: [{table_id}]\n"
                    f"{caption_text}\n"
                    f"Extract EVERY row and EVERY tested cell from the attached table image."
                ),
            },
            image_block,
        ]
        try:
            table_res: TableExtractionResult = retry_on_external_error(
                table_controller.extract_peptides,
                table_prompt_blocks,
                stream=True,
                filename=filename,
                max_retries=max_retries,
                retry_delay=retry_delay,
                context=f"extraction table [{table_id}] {filename}",
            )
        except Exception as e:
            logger.error(f"Failed to extract table [{table_id}] for {filename}: {e}")
            raise
        all_entries.extend(merge_table_result_with_shared(table_res, shared, table_id))

    # Pass 2B: Extract any cropped figure images (phase diagrams, sweeps, vial grids)
    if figure_items:
        figure_controller = LLMController(
            provider=table_provider or provider,
            model_name=table_model_name or model_name,
            structured_output=TableExtractionResult,
            system_prompt=(
                "You are an expert scientific curator extracting experimental peptide "
                "self-assembly data from a single scientific figure (e.g., 2D phase diagram, "
                "pH/concentration/temperature/solvent-ratio sweep, or vial-inversion grid). "
                "Extract every distinct tested (peptide, condition) point visible in the figure, "
                "including non-assembling ('none') regions/markers. If the figure is purely a "
                "micrograph (TEM/SEM/AFM), spectrum (CD/FTIR/NMR/XRD), or schematic with no new "
                "condition-level assembly data points, return rows: []."
            ),
        )
        for fig_id, caption_text, image_block in figure_items:
            fig_prompt_blocks = [
                {
                    "type": "text",
                    "text": (
                        f"Paper-wide shared conditions (merged automatically; only override if this figure differs):\n"
                        f"{shared_summary}\n\n"
                        f"Figure location ID: [{fig_id}]\n"
                        f"{caption_text}\n"
                        f"Extract all experimental assembly/non-assembly condition points shown in this figure. "
                        f"Return an empty rows list if the figure contains no new condition data."
                    ),
                },
                image_block,
            ]
            try:
                fig_res: TableExtractionResult = retry_on_external_error(
                    figure_controller.extract_peptides,
                    fig_prompt_blocks,
                    stream=True,
                    filename=filename,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    context=f"extraction figure [{fig_id}] {filename}",
                )
            except Exception as e:
                logger.error(f"Failed to extract figure [{fig_id}] for {filename}: {e}")
                raise
            all_entries.extend(merge_table_result_with_shared(fig_res, shared, fig_id))

    return PeptideExtractionResult(entries=all_entries)


def run_extraction(
    paper_content: Union[str, bytes, Path, list[dict[str, Any]]],
    provider: str = "google_genai",
    model_name: str = "gemma-4-31b-it",
    filename: str = "paper.pdf",
    system_prompt: Optional[str] = None,
    factored: bool = False,
    table_provider: Optional[str] = None,
    table_model_name: Optional[str] = None,
    agentic: bool = False,
    reviewer_provider: Optional[str] = None,
    reviewer_model_name: Optional[str] = None,
    max_refinement_rounds: int = 1,
    agentic_config: Optional[Any] = None,
    max_retries: int = 0,
    retry_delay: float = 2.0,
) -> PeptideExtractionResult:
    """Run peptide extraction on parsed text, multimodal content blocks, or raw PDF bytes/Path."""
    if agentic:
        from peptide_extractor.agentic.orchestrator import run_agentic_extraction

        return run_agentic_extraction(
            paper_content,
            provider=provider,
            model_name=model_name,
            filename=filename,
            table_provider=table_provider,
            table_model_name=table_model_name,
            reviewer_provider=reviewer_provider,
            reviewer_model_name=reviewer_model_name,
            max_refinement_rounds=max_refinement_rounds,
            config=agentic_config,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    if factored and isinstance(paper_content, list):
        return run_factored_extraction(
            paper_content,
            provider=provider,
            model_name=model_name,
            filename=filename,
            table_provider=table_provider,
            table_model_name=table_model_name,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )

    is_pdf = isinstance(paper_content, (bytes, Path))
    resolved_prompt = system_prompt or (
        PDF_EXTRACTION_SYSTEM_PROMPT if is_pdf else EXTRACTION_SYSTEM_PROMPT_V6
    )

    model = LLMController(
        provider=provider,
        model_name=model_name,
        structured_output=PeptideExtractionResult,
        system_prompt=resolved_prompt,
    )

    return retry_on_external_error(
        model.extract_peptides,
        paper_content,
        stream=True,
        filename=filename,
        max_retries=max_retries,
        retry_delay=retry_delay,
        context=f"extraction {filename}",
    )
