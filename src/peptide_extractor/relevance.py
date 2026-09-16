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

SYSTEM_PROMPT = """You are an expert in scientific paper reading and computational biology.
Your task is to determine whether a given scientific paper is relevant for an information extraction pipeline on 4-20mer peptide self-assembly research based only on its title and abstract.

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


class RelevanceResult(BaseModel):
    detected_clues: List[str] = Field(
        default_factory=list,
        description="Specific experimental techniques, peptide identities, or conditions mentioned in the abstract.",
    )
    explanation: str = Field(
        description="Justification focusing on experimental conditions mentioned or hinted upon.",
    )
    is_relevant: bool = Field(
        description="True if experimental data is potentially reported or hinted at in the fulltext.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0.",
    )


def get_chat_model(
    provider: str = "google_genai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a provider-agnostic LangChain BaseChatModel.

    Supported providers:
    - 'google_genai' / 'google': Google AI Studio models (e.g. 'gemma-4-31b-it', 'gemini-3.5-flash', etc.)
    - 'ollama': Ollama local models (e.g. 'gemma2:27b', 'llama3.1:8b')
    - 'open_router' / 'openrouter': OpenRouter models (e.g. 'google/gemma-4-31b-it', 'nvidia/nemotron-3.5-lightning:free')
    """
    provider_clean = provider.lower().strip()

    if provider_clean in ("google_genai", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_model = model_name or "gemma-4-31b-it"
        resolved_api_key = (
            api_key
            or os.getenv("GOOGLE_AI_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )

        if resolved_api_key and resolved_api_key.strip():
            clean_key = resolved_api_key.strip()
            os.environ["GOOGLE_API_KEY"] = clean_key
            os.environ["GEMINI_API_KEY"] = clean_key
            kwargs["google_api_key"] = clean_key
        else:
            raise ValueError(
                "Google API Key not found!"
            )

        model_kwargs: Dict[str, Any] = {"model": target_model, "temperature": temperature, **kwargs}
        return ChatGoogleGenerativeAI(**model_kwargs)

    elif provider_clean in ("ollama", "langchain_ollama"):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback

        target_model = model_name or "gemma2:27b" # will be updated later
        return ChatOllama(model=target_model, temperature=temperature, **kwargs)

    elif provider_clean in ("open_router", "openrouter", "open-router"):
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError:
            raise ImportError(
                "OpenRouter LangChain package not installed"
            )

        resolved_api_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPEN_ROUTER_API_KEY")
        )
        if resolved_api_key and resolved_api_key.strip():
            clean_key = resolved_api_key.strip()
            os.environ["OPENROUTER_API_KEY"] = clean_key
            kwargs["openrouter_api_key"] = clean_key
        else: 
            raise ValueError("OpenRouter API key not found! Set OPENROUTER_API_KEY or pass api_key.")

        target_model = model_name or "google/gemma-4-31b-it"
        return ChatOpenRouter(model=target_model, temperature=temperature, **kwargs)

    else:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            "Supported options are 'google_genai', 'ollama', 'open_router'"
        )


class PaperRelevanceClassifier:
    """Classifier for evaluating paper abstract relevance using LangChain chat models."""

    def __init__(
        self,
        llm: Optional[BaseChatModel] = None,
        provider: str = "google_genai",
        model_name: Optional[str] = None,
        **kwargs: Any,
    ):
        self.provider = provider if isinstance(provider, str) else "google_genai"
        if isinstance(model_name, str) and model_name.strip():
            self.model_name = model_name.strip()
        elif llm is not None and isinstance(getattr(llm, "model_name", None), str):
            self.model_name = getattr(llm, "model_name")
        elif llm is not None and isinstance(getattr(llm, "model", None), str):
            self.model_name = getattr(llm, "model")
        elif llm is not None:
            self.model_name = "mock-model"
        else:
            if self.provider.lower().strip() in ("ollama", "langchain_ollama"):
                self.model_name = "gemma2:27b"
            elif self.provider.lower().strip() in ("open_router", "openrouter", "open-router"):
                self.model_name = "google/gemma-4-31b-it"
            else:
                self.model_name = "gemma-4-31b-it"

        if llm is not None:
            self.llm = llm
        else:
            self.llm = get_chat_model(provider=provider, model_name=self.model_name, **kwargs)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ])
        try:
            if provider.lower().strip() in ("open_router", "openrouter", "open-router"): # seems like open router langchain doesn't support json_mode 
                self.structured_llm = self.llm.with_structured_output(RelevanceResult)
            else:
                self.structured_llm = self.llm.with_structured_output(RelevanceResult, method="json_mode")
        except Exception:
            self.structured_llm = self.llm.with_structured_output(RelevanceResult)

    def classify_paper(
        self,
        title: Union[str, PaperMetadata],
        abstract: Optional[str] = None,
    ) -> RelevanceResult:
        """Classify paper relevance based on title and abstract text, or a PaperMetadata object."""
        if isinstance(title, PaperMetadata):
            paper_title = title.title
            abstract_text = title.abstract
        else:
            paper_title = title
            abstract_text = abstract

        clean_abstract = abstract_text if abstract_text and abstract_text.strip() else "No abstract provided."
        prompt_val = self.prompt.invoke({"title": paper_title, "abstract": clean_abstract})
        res = self.structured_llm.invoke(prompt_val)
        if isinstance(res, dict):
            return RelevanceResult(**res)
        return res # type: ignore

    def classify_metadata(self, paper: PaperMetadata) -> RelevanceResult:
        """Classify a PaperMetadata object from the database."""
        return self.classify_paper(paper)


def evaluate_paper_relevance(
    paper: PaperMetadata,
    classifier: Optional[PaperRelevanceClassifier] = None,
    session: Optional[Session] = None,
    update_status: bool = True,
) -> tuple[PaperRelevance, RelevanceResult]:
    """Evaluate relevance for a single PaperMetadata record.

    Runs LLM classification, inserts a new PaperRelevance record into the database
    (supporting multiple evaluation runs per paper metadata object), and optionally
    updates paper.status to USABLE or DISCARD.
    """
    if classifier is None:
        classifier = PaperRelevanceClassifier()

    if paper.id is None:
        raise ValueError("PaperMetadata object must have a non-None primary key id.")

    
    rel_result = classifier.classify_paper(paper)

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
        session=session,
    )

    if update_status:
        new_status = PaperStatus.USABLE if rel_result.is_relevant else PaperStatus.DISCARD
        update_paper_status(paper.id, new_status, session=session)

    return relevance_record, rel_result

def screen_papers(
    batch_size: Optional[int] = None,
    classifier: Optional[PaperRelevanceClassifier] = None,
    session: Optional[Session] = None,
    delay_seconds: float = 2.0,
    all_papers: bool = False,
    golden_dataset: bool = False,
) -> List[tuple[PaperMetadata, RelevanceResult]]:
    """Screen papers given conditions for relevance
    """
    if classifier is None:
        classifier = PaperRelevanceClassifier()

    if golden_dataset:
        papers_to_screen = get_paper_metadatas(limit=batch_size, session=session, dataset_subset=DatasetSubset.GOLDEN_BENCHMARK)
    elif all_papers:
        papers_to_screen = get_paper_metadatas(limit=batch_size, session=session)
    else:
        papers_to_screen = get_paper_metadatas(
            status=PaperStatus.NOT_PROCESSED, limit=batch_size, session=session
        )

    results: List[tuple[PaperMetadata, RelevanceResult]] = []

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
            )
            results.append((paper, rel_result))
            print(f"Paper {paper_id} processed")
            logger.info(f"Paper with id {paper_id} is processed")
        except Exception as e:
            if session is not None:
                session.rollback()
            print(f"Skipping paper ID {paper_id} due to API evaluation error: {e}")
            continue
        finally:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return results