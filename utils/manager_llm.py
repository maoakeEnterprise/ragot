"""Answer generation with Qwen3-0.6B and lookup of the chunk texts."""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedTokenizerBase,
    BatchEncoding
    )
from utils import DataManager, ChunkIndex, MinimalSource


class ManagerLLM:
    """Generate answers grounded in the retrieved sources.

    The model is loaded lazily, on the first call to :meth:`generate`.

    Attributes:
        model_name: Hugging Face name of the model.
        tokenizer: Model tokenizer, None until the model is loaded.
        model: Causal language model, None until it is loaded.
        device: ``"cuda"`` if a GPU is available, ``"cpu"`` otherwise.
        sys_prompt: System prompt asking for grounded, concise answers.
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Initialize the manager without loading the model.

        Args:
            model_name: Hugging Face name of the model.
        """
        self.model_name = model_name
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: GenerationMixin | None = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sys_prompt = (
            "You are an assistant answering questions about the "
            "vLLM codebase. "
            "Answer ONLY using the provided context. "
            "If the context does not contain the answer, say "
            "\"I don't know\". "
            "Be concise."
        )

    def _load(self) -> None:
        """Load the tokenizer and the model if not already loaded.

        The model uses bfloat16 on GPU and float32 on CPU.
        """
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)  # type: ignore[arg-type]
        self.model.eval()

    def _build_context(self, sources: list[MinimalSource], texts: list[str],
                       max_context_tokens: int = 3000) -> str:
        """Build the context passed to the model within a token budget.

        Each snippet is prefixed with its file path. Snippets are added in
        rank order and skipped when they would exceed the budget.

        Args:
            sources: Retrieved sources, best first.
            texts: Text of each source, in the same order.
            max_context_tokens: Maximum number of tokens of the context.

        Returns:
            The context string.
        """
        early_context: list[str] = []
        used_token = 0
        assert self.tokenizer is not None

        for source, text in zip(sources, texts):
            details = f"[File: {source.file_path}] \n{text}\n"
            nb_tokens = len(
                self.tokenizer(
                    details,
                    add_special_tokens=False
                )["input_ids"]
            )
            if used_token + nb_tokens > max_context_tokens:
                continue
            early_context.append(details)
            used_token += nb_tokens
        return "\n".join(early_context)

    def generate(self, query: str, sources: list[MinimalSource],
                 texts: list[str], max_new_tokens: int = 384) -> str:
        """Answer a question from the retrieved sources.

        Uses the chat template with thinking disabled and greedy decoding.

        Args:
            query: The user question.
            sources: Retrieved sources, best first.
            texts: Text of each source, in the same order.
            max_new_tokens: Maximum number of generated tokens.

        Returns:
            The generated answer, without leading or trailing whitespace.
        """
        self._load()
        assert self.model is not None and self.tokenizer is not None
        context = self._build_context(sources, texts)
        message = [
            {"role": "system", "content": self.sys_prompt},
            {
                "role": "user",
                "content": f"Context : {context} \nQuestion: {query}"
            }
        ]
        pre_inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        )
        assert isinstance(pre_inputs, BatchEncoding)
        inputs = pre_inputs.to(self.device)
        with torch.inference_mode():
            output = self.model.generate(  # type: ignore[misc]
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        input_len = inputs["input_ids"].shape[1]
        answer_id = output[0][input_len:]
        res = self.tokenizer.decode(
            answer_id,
            skip_special_tokens=True
        )
        assert isinstance(res, str)
        return res.strip()


class ChunkLookUp:
    """Find the text of a source from its location.

    Attributes:
        contents: Maps ``(file_path, first_index, last_index)`` to the
            chunk text.
    """

    def __init__(self, index_dir: str) -> None:
        """Load every chunk of the index into a lookup table.

        Args:
            index_dir: Directory containing ``chunks.json``.
        """
        chunks: ChunkIndex = DataManager.load_chunk_file(
            index_dir=index_dir
        )
        contents: dict[tuple[str, int, int], str] = {
            (
                c.file_path,
                c.first_character_index,
                c.last_character_index,
            ): c.content
            for c in chunks.chunks
        }
        self.contents = contents

    def get(self, source: MinimalSource) -> str:
        """Return the text of a source.

        Args:
            source: Location of the source.

        Returns:
            The chunk text, or an empty string if the source is unknown.
        """
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index
        )
        return self.contents.get(key, "")
