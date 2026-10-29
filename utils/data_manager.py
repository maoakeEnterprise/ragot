"""Persistence of the chunks and of the BM25 index."""
from pathlib import Path
from utils import ChunkIndex, Tokenizer
import bm25s


class DataManager:
    """Build, save and load the chunks and the BM25 index.

    Attributes:
        tokenizer: Tokenizer applied to every chunk before indexing.
        bm25: BM25 model (``k1=1.5``, ``b=0.75``).
    """

    def __init__(self) -> None:
        """Initialize the tokenizer and an empty BM25 model."""
        self.tokenizer = Tokenizer()
        self.bm25 = bm25s.BM25(k1=1.5, b=0.75)

    @staticmethod
    def init_folder(folder: str = "data/processed/") -> None:
        """Create the output folder if it does not exist.

        Args:
            folder: Directory to create.
        """
        Path(folder).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def register_chunk_file(data: str,
                            index_dir: str = "data/processed/") -> None:
        """Write the serialized chunks to ``chunks.json``.

        Args:
            data: JSON dump of a :class:`ChunkIndex`.
            index_dir: Directory where ``chunks.json`` is written.
        """
        Path(index_dir, "chunks.json").write_text(data)

    @staticmethod
    def load_chunk_file(index_dir: str = "data/processed/"
                        ) -> ChunkIndex:
        """Load the chunks from ``chunks.json``.

        Args:
            index_dir: Directory containing ``chunks.json``.

        Returns:
            The validated chunk index.

        Raises:
            FileNotFoundError: If ``chunks.json`` does not exist.
            pydantic.ValidationError: If the file is not a valid index.
        """
        return ChunkIndex.model_validate_json(
            Path(index_dir, "chunks.json").read_text())

    @staticmethod
    def load_index_file(path: str = "data/processed/bm25") -> bm25s.BM25:
        """Load a saved BM25 index.

        Args:
            path: Directory of the saved BM25 index.

        Returns:
            The loaded BM25 model.
        """
        return bm25s.BM25.load(path)

    def save_index_data(self, chunk_index: ChunkIndex, index_dir: str
                        ) -> None:
        """Tokenize every chunk, build the BM25 index and save it.

        The index is saved in ``<index_dir>/bm25``. Its document ids follow
        the order of ``chunk_index.chunks``.

        Args:
            chunk_index: Chunks to index.
            index_dir: Directory where the index is saved.
        """
        self.bm25.index([
            self.tokenizer.tokenize(chunk.content)
            for chunk in chunk_index.chunks
        ])
        self.bm25.save(str(Path(index_dir, "bm25")))

    def get_tokenizer(self) -> Tokenizer:
        """Return the tokenizer used for indexing.

        Returns:
            The tokenizer instance.
        """
        return self.tokenizer
