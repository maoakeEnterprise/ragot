from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.models import Chunk


class Chunker:

    def __init__(self, path: str, scope: list[str]) -> None:
        self.path: Path = Path(path)
        self.scope_files = scope

    def get_files(self, path: str | None = None) -> list[Path]:
        list_path: list[Path] = []
        if path is not None:
            path_p = Path(path)
            list_path = self._algo_files(path_p)
        else:
            list_path = self._algo_files(self.path)
        return list_path

    def _algo_files(self, path: Path) -> list[Path]:
        list_path: list[Path] = []
        for path_file in path.rglob("*"):
            if path_file.is_file() and self._in_scope_files(path_file):
                list_path.append(path_file)
        return list_path

    def _in_scope_files(self, path: Path) -> bool:
        extent = f"{path}".split(".")[-1]
        return extent in self.scope_files

    def _init_chunk_python(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=2000,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _init_chunk_markdown(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=2000,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _init_chunk_txt(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            add_start_index=True
        )
        return splitter

    def _get_extent(self, path: Path) -> str:
        return (f"{path}").split(".")[-1]

    def _get_chunk_list(self, docs: list[Document], path: str) -> list[Chunk]:
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

    def chunk_files(self, list_path: list[Path]) -> list[Chunk]:
        py_splitter = self._init_chunk_python()
        md_splitter = self._init_chunk_markdown()
        txt_splitter = self._init_chunk_txt()
        list_chunk: list[Chunk] = []
        for path in list_path:
            with path.open("r") as f:
                content = f.read()
                if self._get_extent(path) == 'py':
                    tmp = py_splitter.create_documents([content])
                elif self._get_extent(path) == 'md':
                    tmp = md_splitter.create_documents([content])
                else:
                    tmp = txt_splitter.create_documents([content])
                list_chunk.extend(self._get_chunk_list(tmp, f"{path}"))
        return list_chunk
