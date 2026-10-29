"""File discovery and chunking of the corpus."""
from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.models import Chunk
from tqdm import tqdm


class Chunker:
    """Find the corpus files and split them into chunks.

    Python files, Markdown files and plain text files each use their own
    splitting strategy.

    Attributes:
        path: Root directory of the corpus.
        scope_files: File extensions (without the dot) to index.
    """

    def __init__(self, path: str, scope: list[str]) -> None:
        """Initialize the chunker.

        Args:
            path: Root directory of the corpus.
            scope: File extensions (without the dot) to index,
                e.g. ``["py", "md", "txt"]``.
        """
        self.path: Path = Path(path)
        self.scope_files = scope

    def get_files(self, path: str | None = None) -> list[Path]:
        """List recursively every in-scope file.

        Args:
            path: Directory to scan. Defaults to the corpus root.

        Returns:
            The paths of every file whose extension is in scope.
        """
        list_path: list[Path] = []
        if path is not None:
            path_p = Path(path)
            list_path = self._algo_files(path_p)
        else:
            list_path = self._algo_files(self.path)
        return list_path

    def chunk_files(self, list_path: list[Path], max_chunk: int
                    ) -> list[Chunk]:
        """Split every file into chunks of at most ``max_chunk`` characters.

        ``.py`` files use the Python splitter, ``.md`` files the Markdown
        splitter and the other files the plain text splitter.

        Args:
            list_path: Files to chunk.
            max_chunk: Maximum number of characters per chunk.

        Returns:
            Every chunk with its file path, character range and content.
        """
        py_splitter = self._init_chunk_python(max_chunk=max_chunk)
        md_splitter = self._init_chunk_markdown(max_chunk=max_chunk)
        txt_splitter = self._init_chunk_txt(max_chunk=max_chunk)
        list_chunk: list[Chunk] = []
        for path in tqdm(list_path):
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if self._get_extent(path) == 'py':
                    tmp = py_splitter.create_documents([content])
                elif self._get_extent(path) == 'md':
                    tmp = md_splitter.create_documents([content])
                else:
                    tmp = txt_splitter.create_documents([content])
                list_chunk.extend(self._get_chunk_list(tmp, f"{path}"))
        return list_chunk

    def _algo_files(self, path: Path) -> list[Path]:
        """Walk ``path`` recursively and keep the in-scope files.

        Args:
            path: Directory to scan.

        Returns:
            The in-scope files found under ``path``.
        """
        list_path: list[Path] = []
        for path_file in path.rglob("*"):
            if path_file.is_file() and self._in_scope_files(path_file):
                list_path.append(path_file)
        return list_path

    def _in_scope_files(self, path: Path) -> bool:
        """Tell whether a file extension is in scope.

        Args:
            path: File to check.

        Returns:
            True if the file extension is in ``scope_files``.
        """
        extent = f"{path}".split(".")[-1]
        return extent in self.scope_files

    def _init_chunk_python(self, max_chunk: int
                           ) -> RecursiveCharacterTextSplitter:
        """Build the splitter for Python source code.

        It splits first on ``class`` and ``def`` boundaries, then on blank
        lines and lines.

        Args:
            max_chunk: Maximum number of characters per chunk.

        Returns:
            The configured Python splitter.
        """
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=max_chunk,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _init_chunk_markdown(self, max_chunk: int
                             ) -> RecursiveCharacterTextSplitter:
        """Build the splitter for Markdown documents.

        It splits first on headings, then on code blocks and paragraphs.

        Args:
            max_chunk: Maximum number of characters per chunk.

        Returns:
            The configured Markdown splitter.
        """
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=max_chunk,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _init_chunk_txt(self, max_chunk: int
                        ) -> RecursiveCharacterTextSplitter:
        """Build the splitter for plain text files.

        Args:
            max_chunk: Maximum number of characters per chunk.

        Returns:
            The configured plain text splitter.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _get_extent(self, path: Path) -> str:
        """Return the extension of a file, without the dot.

        Args:
            path: File path.

        Returns:
            The text after the last ``.`` of the path.
        """
        return (f"{path}").split(".")[-1]

    def _get_chunk_list(self, docs: list[Document], path: str) -> list[Chunk]:
        """Convert LangChain documents into :class:`Chunk` models.

        Args:
            docs: Documents produced by a splitter, with a ``start_index``
                in their metadata.
            path: Path of the file the documents come from.

        Returns:
            One chunk per document, with its exact character range.
        """
        chunk_list: list[Chunk] = []
        for doc in docs:
            index_start = doc.metadata["start_index"]
            chunk = Chunk(file_path=path,
                          first_character_index=index_start,
                          last_character_index=(index_start
                                                + len(doc.page_content)),
                          content=doc.page_content
                          )
            chunk_list.append(chunk)
        return chunk_list
