from utils import (Chunker, ChunkIndex, DataManager, Retriever, RagDataset,
                   MinimalSearchResults, StudentSearchResults)
from pathlib import Path
from tqdm import tqdm


class Rag:
    def __init__(self, index_dir: str = "data/processed/"):
        self.index_dir = index_dir

    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str = "data/raw/",
            output_dir: str = "data/processed/"
    ) -> None:

        if (not isinstance(max_chunk_size, int)
                or isinstance(max_chunk_size, bool)
                or not 201 <= max_chunk_size <= 2000):
            raise ValueError("max_chunk_size should be an integer "
                             "between [201, 2000]")
        corpus_dir = Path(raw_dir, "vllm-0.10.1")
        if not corpus_dir.is_dir():
            raise FileNotFoundError(f"The directory {corpus_dir} does not "
                                    "exist")

        scope = [
                'py',
                'md',
                'txt'
            ]
        data_m = DataManager()
        indexer = Chunker(path=str(corpus_dir), scope=scope)
        files = indexer.get_files()
        if len(files) == 0:
            raise ValueError(f"No .py/.md/.txt files found in {corpus_dir}")
        chunk_index = ChunkIndex(
            chunks=indexer.chunk_files(
                list_path=files,
                max_chunk=max_chunk_size
                ))
        data_m.init_folder(folder=output_dir)
        data_m.register_chunk_file(
            data=chunk_index.model_dump_json(),
            index_dir=output_dir
        )
        data_m.save_index_data(
            chunk_index=chunk_index, index_dir=output_dir
        )

    @staticmethod
    def _check_k(k: int) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k should be an integer upper to 0")

    def search(self, query: str, k: int = 5) -> None:

        self._check_k(k)
        retriever = Retriever(index_dir=self.index_dir)
        source = retriever.search(query=query, k=k)
        for ms in source:
            print(f"{ms.file_path} [{ms.first_character_index}:"
                  f"{ms.last_character_index}]")

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 5,
            save_directory: str = "data/output/search_results"
    ) -> None:
        self._check_k(k)
        retriever = Retriever(index_dir=self.index_dir)
        if k > len(retriever.chunk_index.chunks):
            raise ValueError("k should not be larger than the number of "
                             f"chunks ({len(retriever.chunk_index.chunks)})")
        data_set = RagDataset.model_validate_json(
            Path(dataset_path).read_text())
        results: list[MinimalSearchResults] = []

        for q in tqdm(data_set.rag_questions):
            try:
                sources = retriever.search(query=q.question, k=k)
            except ValueError:
                sources = []
            msr = MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources
            )
            results.append(msr)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output = StudentSearchResults(
            search_results=results,
            k=k
        )
        (save_dir / Path(dataset_path).name).write_text(
            output.model_dump_json(indent=2))

    def answer(self, query: str, k: int = 5) -> None:
        pass

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> None:
        pass

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str
    ) -> None:
        pass
