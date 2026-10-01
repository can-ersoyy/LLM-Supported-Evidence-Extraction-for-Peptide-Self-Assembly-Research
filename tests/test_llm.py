from unittest.mock import MagicMock
import pytest

from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.llm import LLMController, get_chat_model
from peptide_extractor.schemas import RelevanceResult


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
