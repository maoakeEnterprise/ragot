from pathlib import Path
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Indexer:

    def __init__(self, path: str, scope: list[str]) -> None:
        self.path: Path = Path(path)
        self.scope_files = scope

    def get_files(self, path: str | None = None) -> None:
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
            chunk_size=300,
            chunk_overlap=100
        )
        return splitter

    def _init_chunk_markdown(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=300,
            chunk_overlap=100
        )
        return splitter

    def _init_chunk_txt(self) -> RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=100
        )
        return splitter

    def _get_extent(self, path: Path) -> str:
        return (f"{path}").split(".")[-1]

    def chunk_files(self, list_path: list[Path]) -> list[Document]:
        py_splitter = self._init_chunk_python()
        md_splitter = self._init_chunk_markdown()
        txt_splitter = self._init_chunk_txt()
        list_doc: list[Document] = []
        for path in list_path:
            content = path.read_text()
            if self._get_extent(path) == 'py':
                list_doc.append(py_splitter.create_documents([content]))
            elif self._get_extent(path) == 'md':
                list_doc.append(md_splitter.create_documents([content]))
            else:
                list_doc.append(txt_splitter.create_documents([content]))
        return list_doc
