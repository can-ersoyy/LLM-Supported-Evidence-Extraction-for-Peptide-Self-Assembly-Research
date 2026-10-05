from peptide_extractor.db import PeptideEntry
from typing import List
import os
from typing import Any, Dict, Optional, Type, Union

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from peptide_extractor.db.paper_metadata import PaperMetadata

load_dotenv(find_dotenv(usecwd=True))

DEFAULT_MODELS: Dict[str, str] = {
    "google": "gemma-4-31b-it",
    "google_genai": "gemma-4-31b-it",
    "ollama": "gemma2:27b",
    "openrouter": "google/gemma-4-31b-it",
    "open_router": "google/gemma-4-31b-it",
}


def get_chat_model(
    provider: str = "google_genai",
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a provider-agnostic LangChain BaseChatModel."""
    provider_clean = provider.lower().strip()

    if provider_clean in ("google_genai", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_model = model_name or DEFAULT_MODELS["google"]
        resolved_key = (
            api_key
            or os.getenv("GOOGLE_AI_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
        )
        if not resolved_key:
            raise ValueError("Google API Key not found.")
        clean_key = resolved_key.strip()
        os.environ["GOOGLE_API_KEY"] = clean_key
        return ChatGoogleGenerativeAI(
            model=target_model,
            temperature=temperature,
            google_api_key=clean_key,
            **kwargs,
        )

    elif provider_clean in ("ollama", "langchain_ollama"):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama

        target_model = model_name or DEFAULT_MODELS["ollama"]
        return ChatOllama(model=target_model, temperature=temperature, **kwargs)

    elif provider_clean in ("open_router", "openrouter", "open-router"):
        try:
            from langchain_openrouter import ChatOpenRouter
        except ImportError:
            raise ImportError("OpenRouter LangChain package not installed.")

        resolved_key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPEN_ROUTER_API_KEY")
        )
        if not resolved_key:
            raise ValueError("OpenRouter API key not found.")
        clean_key = resolved_key.strip()
        os.environ["OPENROUTER_API_KEY"] = clean_key
        return ChatOpenRouter(
            model=model_name or DEFAULT_MODELS["openrouter"],
            temperature=temperature,
            openrouter_api_key=clean_key,
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider '{provider}'.")


class LLMController:
    """Simple controller managing LangChain models and task methods."""

    def __init__(
        self,
        provider: str = "google_genai",
        model_name: Optional[str] = None,
        structured_output: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.structured_output = structured_output
        self.model_name = model_name or (
            getattr(llm, "model_name", None)
            or getattr(llm, "model", None)
            or DEFAULT_MODELS.get(provider.lower().strip(), "gemma-4-31b-it")
        )

        self.llm = llm or get_chat_model(
            provider=provider,
            model_name=self.model_name,
            temperature=temperature,
            api_key=api_key,
            **kwargs,
        )

        if self.structured_output:
            try:
                self.structured_llm = self.llm.with_structured_output(self.structured_output, method="json_mode")
            except Exception:
                self.structured_llm = self.llm.with_structured_output(self.structured_output)
        else:
            self.structured_llm = self.llm

    def classify_paper(
        self,
        title: Union[str, PaperMetadata],
        abstract: Optional[str] = None,
    ) -> Any:
        """Classify relevance of a paper by title and abstract or PaperMetadata."""
        if isinstance(title, PaperMetadata):
            paper_title = title.title or ""
            abstract_text = title.abstract or "No abstract provided."
        else:
            paper_title = title
            abstract_text = abstract if abstract and abstract.strip() else "No abstract provided."

        user_content = (
            f"Here is the title and abstract of a paper:\n\n"
            f"Title: {paper_title}\n\n"
            f"Abstract: {abstract_text}"
        )

        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_content))

        res = self.structured_llm.invoke(messages)
        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res

    def extract_peptides(self, paper_content: str) -> Any:
        """Run extraction on paper content."""
        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=f"Fulltext:\n\n{paper_content}"))

        res = self.structured_llm.invoke(messages)
        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res

