from peptide_extractor.db.peptide_entry import PeptideExtractionResult
from peptide_extractor.schemas import RelevanceResult
from unittest.mock import MagicMock
import pytest

from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.llm import LLMController, get_chat_model


@pytest.fixture
def mock_chat_model():
    """Mock LangChain chat model."""
    llm = MagicMock()
    structured = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm, structured


def test_llm_controller_init(mock_chat_model):
    llm, _ = mock_chat_model
    controller = LLMController(
        llm=llm,
        provider="google_genai",
        structured_output=RelevanceResult,
        system_prompt="Test system prompt",
    )
    assert controller.provider == "google_genai"
    assert controller.system_prompt == "Test system prompt"
    assert controller.structured_output == RelevanceResult


def test_llm_controller_classify_paper_title_abstract(mock_chat_model):
    llm, structured = mock_chat_model
    sample = RelevanceResult(
        detected_clues=["nanofiber"],
        explanation="Assembles into nanofibers.",
        is_relevant=True,
        confidence=0.9,
    )
    structured.invoke.return_value = sample

    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
        system_prompt="System prompt",
    )

    result = controller.classify_paper(
        title="Peptide Study", abstract="Forms nanofibers."
    )
    assert result.is_relevant is True
    assert result.confidence == 0.9
    assert result.detected_clues == ["nanofiber"]
    structured.invoke.assert_called_once()


def test_llm_controller_classify_paper_metadata(mock_chat_model):
    llm, structured = mock_chat_model
    sample = RelevanceResult(
        detected_clues=["hydrogel"],
        explanation="Forms hydrogel.",
        is_relevant=True,
        confidence=0.88,
    )
    structured.invoke.return_value = sample

    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
    )

    paper = PaperMetadata(title="Hydrogel Paper", abstract="Forms firm hydrogel.")
    result = controller.classify_paper(paper)
    assert result.is_relevant is True
    assert result.confidence == 0.88


def test_llm_controller_dict_result_conversion(mock_chat_model):
    llm, structured = mock_chat_model
    structured.invoke.return_value = {
        "detected_clues": ["amyloid"],
        "explanation": "Amyloid fibril formation.",
        "is_relevant": True,
        "confidence": 0.92,
    }

    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
    )

    result = controller.classify_paper(
        title="Amyloid Study", abstract="Forms amyloid fibrils."
    )
    assert isinstance(result, RelevanceResult)
    assert result.is_relevant is True
    assert result.confidence == 0.92


def test_get_chat_model_missing_keys(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_AI_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Google API Key not found"):
        get_chat_model(provider="google_genai")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPEN_ROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenRouter API key not found"):
        get_chat_model(provider="open_router")


def test_get_chat_model_ollama():
    chat_ollama = get_chat_model(provider="ollama", model_name="llama3.2")
    assert getattr(chat_ollama, "model", None) == "llama3.2"


def test_get_chat_model_invalid_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_chat_model(provider="invalid_provider")


def test_extract_peptides_with_curly_braces(mock_chat_model):
    """Ensure extract_peptides handles scientific text with LaTeX formulas and curly braces without template error."""
    llm, structured = mock_chat_model
    structured.invoke.return_value = {"entries": []}

    controller = LLMController(
        llm=llm,
        structured_output=PeptideExtractionResult,
        system_prompt="Extract peptides with {schema}",
    )

    paper_content_with_latex = (
        r"Peptide was dissolved at 10 mg/mL. Equation: $$R\hskip 2pt{\mathrm{ = \pi }}r_{\mathrm{B}}^0\sqrt {\frac{{m_0/M}}{{\fra...$$ "
        r"Using formula {C6H12O6} and ratio {1:1} with \documentclass[12pt]{minimal}."
    )
    result = controller.extract_peptides(paper_content_with_latex)
    assert isinstance(result, PeptideExtractionResult)
    assert result.entries == []
    structured.invoke.assert_called_once()
    called_messages = structured.invoke.call_args[0][0]
    assert any(
        "$$R\\hskip 2pt{\\mathrm{ = \\pi }}" in msg.content for msg in called_messages
    )


def test_classify_paper_with_curly_braces(mock_chat_model):
    """Ensure classify_paper handles abstracts with chemical notations or brackets containing curly braces."""
    llm, structured = mock_chat_model
    structured.invoke.return_value = RelevanceResult(
        detected_clues=["nanotube"],
        explanation="Valid",
        is_relevant=True,
        confidence=0.9,
    )

    controller = LLMController(
        llm=llm,
        structured_output=RelevanceResult,
        system_prompt="Classify {relevance}",
    )

    res = controller.classify_paper(
        title="Study of {FF} Nanostructures",
        abstract=r"We observed aggregation with \Delta G^{\circ} = -2.5 {kJ/mol}.",
    )
    assert res.is_relevant is True
    structured.invoke.assert_called_once()


def test_extract_peptides_pdf_bytes(mock_chat_model):
    """Ensure extract_peptides builds a standard LangChain file block for raw PDF bytes."""
    import base64
    from langchain_core.messages import HumanMessage

    llm, structured = mock_chat_model
    structured.invoke.return_value = {"entries": []}

    controller = LLMController(
        llm=llm,
        provider="google_genai",
        structured_output=PeptideExtractionResult,
        system_prompt="Extract from PDF",
    )

    pdf_bytes = b"%PDF-1.4 fake pdf content"
    result = controller.extract_peptides(pdf_bytes, filename="table_paper.pdf")

    assert isinstance(result, PeptideExtractionResult)
    called_messages = structured.invoke.call_args[0][0]
    human_msgs = [m for m in called_messages if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 1
    content = human_msgs[0].content
    assert isinstance(content, list)
    file_blocks = [b for b in content if b.get("type") == "file"]
    assert len(file_blocks) == 1
    assert file_blocks[0]["mime_type"] == "application/pdf"
    assert file_blocks[0]["filename"] == "table_paper.pdf"
    assert base64.b64decode(file_blocks[0]["base64"]) == pdf_bytes


def test_extract_peptides_pdf_path(mock_chat_model, tmp_path):
    """Ensure extract_peptides accepts a pathlib.Path to a PDF file."""
    import base64
    from langchain_core.messages import HumanMessage

    llm, structured = mock_chat_model
    structured.invoke.return_value = {"entries": []}

    pdf_path = tmp_path / "sample_table.pdf"
    pdf_bytes = b"%PDF-1.4 sample"
    pdf_path.write_bytes(pdf_bytes)

    controller = LLMController(
        llm=llm,
        provider="google_genai",
        structured_output=PeptideExtractionResult,
    )

    result = controller.extract_peptides(pdf_path)
    assert isinstance(result, PeptideExtractionResult)
    human_msg = structured.invoke.call_args[0][0][0]
    assert isinstance(human_msg, HumanMessage)
    file_block = [b for b in human_msg.content if b.get("type") == "file"][0]
    assert file_block["filename"] == "sample_table.pdf"
    assert base64.b64decode(file_block["base64"]) == pdf_bytes


def test_extract_peptides_pdf_ollama_raises(mock_chat_model):
    """Ensure passing raw PDF bytes to Ollama raises a clear ValueError."""
    llm, _ = mock_chat_model
    controller = LLMController(
        llm=llm,
        provider="ollama",
        structured_output=PeptideExtractionResult,
    )

    with pytest.raises(ValueError, match="Direct PDF input is not supported"):
        controller.extract_peptides(b"%PDF-1.4 fake")


def test_run_extraction_selects_pdf_prompt(monkeypatch):
    """Ensure run_extraction uses PDF_EXTRACTION_SYSTEM_PROMPT for PDF bytes."""
    from peptide_extractor import extraction as extraction_mod

    captured = {}

    class DummyController:
        def __init__(self, provider, model_name, structured_output, system_prompt):
            captured["provider"] = provider
            captured["model_name"] = model_name
            captured["system_prompt"] = system_prompt

        def extract_peptides(self, paper_content, stream=True, filename="paper.pdf"):
            captured["paper_content"] = paper_content
            captured["filename"] = filename
            return PeptideExtractionResult(entries=[])

    monkeypatch.setattr(extraction_mod, "LLMController", DummyController)

    pdf_bytes = b"%PDF-1.4 test"
    res = extraction_mod.run_extraction(
        pdf_bytes,
        provider="google_genai",
        model_name="gemini-3.5-flash-lite",
        filename="my_paper.pdf",
    )
    assert isinstance(res, PeptideExtractionResult)
    assert captured["system_prompt"] == extraction_mod.PDF_EXTRACTION_SYSTEM_PROMPT
    assert captured["paper_content"] == pdf_bytes
    assert captured["filename"] == "my_paper.pdf"


def test_extract_peptides_multimodal_blocks(mock_chat_model):
    """Ensure extract_peptides passes multimodal text+image blocks directly to HumanMessage."""
    from langchain_core.messages import HumanMessage

    llm, structured = mock_chat_model
    structured.invoke.return_value = {"entries": []}

    controller = LLMController(
        llm=llm,
        provider="google_genai",
        structured_output=PeptideExtractionResult,
        system_prompt="Extract from blocks",
    )

    blocks = [
        {"type": "text", "text": "[sec-1/table-0] Table 1\n>>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
    ]
    result = controller.extract_peptides(blocks)

    assert isinstance(result, PeptideExtractionResult)
    called_messages = structured.invoke.call_args[0][0]
    human_msgs = [m for m in called_messages if isinstance(m, HumanMessage)]
    assert len(human_msgs) == 1
    assert human_msgs[0].content == blocks


def test_run_extraction_selects_text_prompt_for_blocks(monkeypatch):
    """Ensure run_extraction uses EXTRACTION_SYSTEM_PROMPT_V6 for multimodal blocks."""
    from peptide_extractor import extraction as extraction_mod

    captured = {}

    class DummyController:
        def __init__(self, provider, model_name, structured_output, system_prompt):
            captured["system_prompt"] = system_prompt

        def extract_peptides(self, paper_content, stream=True, filename="paper.pdf"):
            captured["paper_content"] = paper_content
            return PeptideExtractionResult(entries=[])

    monkeypatch.setattr(extraction_mod, "LLMController", DummyController)

    blocks = [
        {"type": "text", "text": "[sec-1/table-0] Table 1"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
    ]
    res = extraction_mod.run_extraction(blocks)
    assert isinstance(res, PeptideExtractionResult)
    assert captured["system_prompt"] == extraction_mod.EXTRACTION_SYSTEM_PROMPT_V6
    assert captured["paper_content"] == blocks


def test_factored_extraction_splits_and_merges_shared_conditions(monkeypatch):
    """Verify two-pass factored extraction gathers shared conditions once and merges into unrolled table rows."""
    from peptide_extractor import extraction as extraction_mod
    from peptide_extractor.db.peptide_entry import EvidenceSpan, PeptideAssemblyType

    blocks = [
        {
            "type": "text",
            "text": (
                "[sec-0/p-0/s-0] Tripeptides were incubated at 20 °C for 24 h.\n"
                "[sec-1/table-0] Table 1\n"
                "\t>>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<\n"
            ),
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,tbl0"}},
        {
            "type": "text",
            "text": "[sec-2/p-0/s-0] End of results.\n",
        },
    ]

    calls = []

    class DummyFactoredController:
        def __init__(self, provider, model_name, structured_output, system_prompt):
            self.structured_output = structured_output

        def extract_peptides(self, paper_content, stream=True, filename="paper.pdf"):
            calls.append((self.structured_output, paper_content))
            if self.structured_output is extraction_mod.PaperWideExtractionResult:
                return extraction_mod.PaperWideExtractionResult(
                    shared_conditions=extraction_mod.SharedExperimentalConditions(
                        n_terminal="Boc",
                        c_terminal="OCH2Ph",
                        temperature=20.0,
                        incubation_period=1440,
                        detection_method=["Inverted tube"],
                        evidence_span=EvidenceSpan(
                            temperature="[sec-0/p-0/s-0]",
                            incubation_period="[sec-0/p-0/s-0]",
                        ),
                    ),
                    prose_entries=[],
                )
            return extraction_mod.TableExtractionResult(
                rows=[
                    extraction_mod.CompactTableRow(
                        sequence="GGG",
                        conditions=[
                            extraction_mod.CompactTableCondition(
                                solution="hexane", type=PeptideAssemblyType.NONE
                            ),
                            extraction_mod.CompactTableCondition(
                                solution="toluene",
                                type=PeptideAssemblyType.ORGANOGEL,
                                peptide_concentration=10.0,
                            ),
                        ],
                    )
                ]
            )

    monkeypatch.setattr(extraction_mod, "LLMController", DummyFactoredController)

    res = extraction_mod.run_extraction(blocks, factored=True)
    assert len(calls) == 2
    assert calls[0][0] is extraction_mod.PaperWideExtractionResult
    assert isinstance(calls[0][1], str)  # Pass 1 receives prose text only
    assert calls[1][0] is extraction_mod.TableExtractionResult
    assert isinstance(calls[1][1], list)  # Pass 2 receives table image block

    assert len(res.entries) == 2
    e0, e1 = res.entries
    assert e0.sequence == "GGG"
    assert e0.solution == "hexane"
    assert e0.type == PeptideAssemblyType.NONE
    assert e0.temperature == 20.0
    assert e0.incubation_period == 1440
    assert e0.n_terminal == "Boc"
    assert e0.c_terminal == "OCH2Ph"
    assert e0.evidence_span.temperature == "[sec-0/p-0/s-0]"
    assert e0.evidence_span.incubation_period == "[sec-0/p-0/s-0]"
    assert e0.evidence_span.sequence == "[sec-1/table-0]"
    assert e0.evidence_span.solution == "[sec-1/table-0]"

    assert e1.solution == "toluene"
    assert e1.type == PeptideAssemblyType.ORGANOGEL
    assert e1.peptide_concentration == 10.0
    assert e1.evidence_span.peptide_concentration == "[sec-1/table-0]"


def test_factored_extraction_uses_separate_table_provider_and_model(monkeypatch):
    """Verify Pass 2 can use a distinct table_provider and table_model_name while Pass 1 uses provider/model_name."""
    from peptide_extractor import extraction as extraction_mod

    blocks = [
        {
            "type": "text",
            "text": "[sec-1/table-0] Table 1\n\t>>> [TABLE IMAGE ATTACHED: sec-1/table-0] <<<\n",
        },
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,tbl0"}},
    ]

    init_calls = []

    class DummyController:
        def __init__(self, provider, model_name, structured_output, system_prompt):
            init_calls.append((structured_output, provider, model_name))
            self.structured_output = structured_output

        def extract_peptides(self, paper_content, stream=True, filename="paper.pdf"):
            if self.structured_output is extraction_mod.PaperWideExtractionResult:
                return extraction_mod.PaperWideExtractionResult()
            return extraction_mod.TableExtractionResult(rows=[])

    monkeypatch.setattr(extraction_mod, "LLMController", DummyController)

    extraction_mod.run_extraction(
        blocks,
        provider="google_genai",
        model_name="gemini-2.5-pro",
        factored=True,
        table_provider="openrouter",
        table_model_name="gemma-4-31b-it",
    )

    assert init_calls == [
        (extraction_mod.PaperWideExtractionResult, "google_genai", "gemini-2.5-pro"),
        (extraction_mod.TableExtractionResult, "openrouter", "gemma-4-31b-it"),
    ]



