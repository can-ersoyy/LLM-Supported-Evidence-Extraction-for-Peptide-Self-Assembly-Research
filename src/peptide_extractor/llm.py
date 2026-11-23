import base64
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from peptide_extractor.db.paper_metadata import PaperMetadata
from peptide_extractor.env import load_env

logger = logging.getLogger(__name__)

load_env()


class StreamingTokenPrinter(BaseCallbackHandler):
    """Callback handler for streaming LLM requests without polluting stdout/logs."""

    def on_llm_start(self, *args, **kwargs) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass

    def on_llm_end(self, *args, **kwargs) -> None:
        pass


def get_chat_model(
    provider: str = "google_genai",
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> BaseChatModel:
    """Instantiate a provider-agnostic LangChain BaseChatModel."""
    provider_clean = provider.lower().strip()

    if provider_clean in ("google_genai", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        target_model = model_name
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

        target_model = model_name
        return ChatOllama(
            base_url=base_url,
            model=target_model,
            temperature=temperature,
            num_ctx=32000,
            num_predict=5000,
            **kwargs,
        )

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
            model=model_name,
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
        model_name: str = "gemini-3.5-flash-lite",
        structured_output: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        llm: Optional[BaseChatModel] = None,
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # used for remote llm connection via ollama
        stream: bool = False,
        **kwargs: Any,
    ):
        self.provider = provider
        self.system_prompt = system_prompt
        self.structured_output = structured_output
        self.model_name = model_name or (
            getattr(llm, "model_name", None) or getattr(llm, "model", None)
        )

        self.llm = llm or get_chat_model(
            provider=provider,
            model_name=self.model_name,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
        self.stream = stream

        if self.structured_output and provider != "ollama":
            try:
                self.structured_llm = self.llm.with_structured_output(
                    self.structured_output, method="json_mode"
                )
            except Exception:
                self.structured_llm = self.llm.with_structured_output(
                    self.structured_output
                )
        elif self.structured_output and provider == "ollama":
            try:
                self.structured_llm = self.llm.with_structured_output(
                    self.structured_output, method="json_schema"
                )
            except Exception:
                self.structured_llm = self.llm.with_structured_output(
                    self.structured_output
                )
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
            abstract_text = (
                abstract if abstract and abstract.strip() else "No abstract provided."
            )

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

    def bulk_classify_paper(
        self, id_title_abstract_pairs: List[Tuple[int, str, str]], stream: bool = True
    ):

        prompt_strings = ""
        for id, title, abstract in id_title_abstract_pairs:
            prompt_strings += "-------------------------------------\n"
            prompt_strings += f"ID: {id}\n"
            prompt_strings += f"Title: {title}\n"
            prompt_strings += f"Abstract: {abstract}\n"
            prompt_strings += "-------------------------------------\n"

        user_content: str = (
            f"Here are title and abstracts of {len(id_title_abstract_pairs)} papers. Work on each pair of title/abstract seperately.",
            prompt_strings,
        )

        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=user_content))

        should_stream = self.stream if stream is None else stream

        if should_stream:
            callbacks = [StreamingTokenPrinter()]
            res = self.structured_llm.invoke(messages, config={"callbacks": callbacks})
            logger.debug(f"LLM output: {res}")
        else:
            res = self.structured_llm.invoke(messages)
            logger.debug(f"LLM output: {res}")

        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res

    def extract_peptides(
        self,
        paper_content: Union[str, bytes, Path, List[Dict[str, Any]]],
        stream: Optional[bool] = None,
        filename: str = "paper.pdf",
    ) -> Any:
        """Run extraction on paper text, multimodal content blocks, or raw PDF bytes/Path."""
        if isinstance(paper_content, Path):
            filename = paper_content.name
            paper_content = paper_content.read_bytes()

        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))

        if isinstance(paper_content, bytes):
            provider_clean = (self.provider or "").lower().strip()
            if provider_clean in ("ollama", "langchain_ollama"):
                raise ValueError(
                    f"Direct PDF input is not supported by provider '{self.provider}'. "
                    "Use 'google_genai' or 'open_router', or pass extracted text instead."
                )
            encoded_pdf = base64.b64encode(paper_content).decode("utf-8")
            messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": "Extract all experimental peptide self-assembly information from the attached PDF document.",
                        },
                        {
                            "type": "file",
                            "base64": encoded_pdf,
                            "mime_type": "application/pdf",
                            "filename": filename,
                        },
                    ]
                )
            )
        elif isinstance(paper_content, list):
            if not paper_content:
                raise ValueError("paper_content blocks list cannot be empty.")
            messages.append(HumanMessage(content=paper_content))
        else:
            messages.append(HumanMessage(content=f"Fulltext:\n\n{paper_content}"))

        should_stream = self.stream if stream is None else stream

        if should_stream:
            callbacks = [StreamingTokenPrinter()]
            res = self.structured_llm.invoke(messages, config={"callbacks": callbacks})
        else:
            res = self.structured_llm.invoke(messages)

        logger.debug(f"LLM output: {res}")
        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res

    def evaluate_faithfullness(self, field_value: str, evidence: str) -> bool:
        """Evaluate if the field value is faithful to the evidence."""
        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(
            HumanMessage(content=f"Field value: {field_value}\n\nEvidence: {evidence}")
        )

        res = self.structured_llm.invoke(messages)
        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res

    def evaluate_groundness(self, field_value_quote_list: list) -> bool:
        """Evaluate if the field value is grounded in the original text."""
        messages: List[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(
            HumanMessage(
                content=f"""Following is a list of three item tuples (field, value, quote). Check if the value for each tuple is grounded in the corresponding quote. Return a list of booleans indicating whether each value is grounded.
        {field_value_quote_list}
        """
            )
        )

        res = self.structured_llm.invoke(messages)
        if isinstance(res, dict) and self.structured_output:
            return self.structured_output(**res)
        return res
