from pathlib import Path
from utils import ChunkIndex
import bm25s


class DataManager:

    @staticmethod
    def init_folder(folder: str = "data/processed") -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def register_chunk_file(data: str,
                            path: str = "data/processed/chunks.json") -> None:
        Path(path).write_text(data)

    @staticmethod
    def load_chunk_file(path: str = "data/processed/chunks.json"
                        ) -> ChunkIndex:
        return ChunkIndex.model_validate_json(Path(path).read_text())

    @staticmethod
    def load_index_file(path: str = "data/processed/bm25") -> bm25s.BM25:
        return bm25s.BM25.load(path)
