from unittest.mock import MagicMock
import pytest

from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.llm import LLMController, get_chat_model
from peptide_extractor.schemas import PeptideExtractionResult, RelevanceResult


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

    result = controller.classify_paper(title="Peptide Study", abstract="Forms nanofibers.")
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

    result = controller.classify_paper(title="Amyloid Study", abstract="Forms amyloid fibrils.")
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
    assert any("$$R\\hskip 2pt{\\mathrm{ = \\pi }}" in msg.content for msg in called_messages)


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

