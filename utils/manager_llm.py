import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationMixin,
    PreTrainedTokenizerBase
    )
from utils import DataManager, ChunkIndex, MinimalSource


class ManagerLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
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
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_context(self, sources: list[MinimalSource], texts: list[str],
                       max_context_tokens: int = 3000) -> str:
        early_context: list[str] = []
        used_token = 0

        for source, text in zip(sources, texts):
            details = f"[File: {source.file_path}] \n{text}\n"
            nb_tokens = len(
                self.tokenizer(
                    details,
                    add_special_tokens=False
                )["input_ids"]
            )
            if used_token + nb_tokens > max_context_tokens:
                break
            early_context.append(details)
            used_token += nb_tokens
        return "\n".join(early_context)

    def generate(self, question: str, sources: list[MinimalSource],
                 texts: list[str], max_new_tokens: int = 384) -> str:
        self._load()
        assert self.model is not None and self.tokenizer is not None
        context = self._build_context(sources, texts)
        message = [
            {"role": "system", "content": self.sys_prompt},
            {
                "role": "user",
                "content": f"content : {context} \nquestion: {question}"
            }
        ]
        inputs = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        input_len = inputs["input_ids"].shape[1]
        answer_id = output[0][input_len:]
        return self.tokenizer.decode(
            answer_id,
            skip_special_tokens=True
        ).strip()


class ChunkLookUp:
    def __init__(self, index_dir: str) -> None:
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
        key = (
            source.file_path,
            source.first_character_index,
            source.last_character_index
        )
        return self.contents[key]
