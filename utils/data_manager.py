from pathlib import Path
from utils import ChunkIndex, Tokenizer
import bm25s


class DataManager:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.bm25 = bm25s.BM25(k1=1.5, b=0.75)

    @staticmethod
    def init_folder(folder: str = "data/processed") -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def register_chunk_file(data: str,
                            index_dir: str = "data/processed") -> None:
        Path(f"{index_dir}/chunks.json").write_text(data)

    @staticmethod
    def load_chunk_file(index_dir: str = "data/processed"
                        ) -> ChunkIndex:
        return ChunkIndex.model_validate_json(
            Path(f"{index_dir}/chunks.json").read_text())

    @staticmethod
    def load_index_file(path: str = "data/processed/bm25") -> bm25s.BM25:
        return bm25s.BM25.load(path)

    def save_index_data(self, chunk_index: ChunkIndex, index_dir: str
                        ) -> None:
        self.bm25.index([
            self.tokenizer.tokenize(chunk.content)
            for chunk in chunk_index.chunks
        ])
        self.bm25.save(f"{index_dir}/bm25")

    def get_tokenizer(self) -> Tokenizer:
        return self.tokenizer
