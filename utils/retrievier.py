from utils import Tokenizer, DataManager, MinimalSource
import bm25s
from pathlib import Path


class Retriever:
    def __init__(self, index_dir: str, tokenizer: Tokenizer | None = None
                 ) -> None:
        path = Path(index_dir, "bm25")
        if not path.exists():
            raise FileNotFoundError("The index did not exist you should try "
                                    "to start the command index:\n"
                                    "uv run python -m src index "
                                    "--max_chunk_size <number>")
        self.bm25 = bm25s.BM25.load(str(path))
        self.chunk_index = DataManager.load_chunk_file(
            index_dir=index_dir)
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        ms_l: list[MinimalSource] = []
        token_query = self.tokenizer.tokenize(query)
        if len(token_query) == 0:
            raise ValueError("The Query is not good")
        res = self.bm25.retrieve([token_query], k=k)
        for id in res.documents[0]:
            chunk = self.chunk_index.chunks[id]
            ms = MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            )
            ms_l.append(ms)
        return ms_l
