"""CLI commands of the RAG system."""
from utils import (Chunker, ChunkIndex, DataManager, Retriever, RagDataset,
                   MinimalSearchResults, StudentSearchResults,
                   ManagerLLM, ChunkLookUp, MinimalAnswer,
                   StudentSearchResultsAndAnswer, AnsweredQuestion,
                   MinimalSource)
from pathlib import Path
from tqdm import tqdm


class Rag:
    """RAG pipeline exposed as CLI commands through Python Fire.

    Attributes:
        index_dir: Directory of the index used by the search and answer
            commands.
    """

    def __init__(self, index_dir: str = "data/processed/"):
        """Initialize the pipeline.

        Args:
            index_dir: Directory of the index (``chunks.json`` and
                ``bm25/``).
        """
        self.index_dir = index_dir

    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str = "data/raw/",
            output_dir: str = "data/processed/"
    ) -> None:
        """Chunk the vLLM corpus and build the BM25 index.

        Args:
            max_chunk_size: Maximum number of characters per chunk,
                between 201 and 2000.
            raw_dir: Directory containing ``vllm-0.10.1/``.
            output_dir: Directory where the index is saved.

        Raises:
            ValueError: If ``max_chunk_size`` is invalid or no file is
                found.
            FileNotFoundError: If the corpus directory does not exist.
        """

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
        print(f" Ingestion complete! Indexed {len(chunk_index.chunks)} chunks "
              f"under {output_dir}")

    @staticmethod
    def _check_k(k: int) -> None:
        """Check that ``k`` is a strictly positive integer.

        Args:
            k: Number of results requested.

        Raises:
            ValueError: If ``k`` is not an integer or is lower than 1.
        """
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k should be an integer upper to 0")

    def search(self, query: str, k: int = 5) -> None:
        """Print the top-k sources for a single query.

        Each line has the form ``file_path [first:last]``.

        Args:
            query: The question.
            k: Number of sources to return.

        Raises:
            ValueError: If ``k`` is invalid or the query has no usable
                token.
            FileNotFoundError: If the index does not exist.
        """

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
        """Search every question of a dataset and save the results.

        The output is a :class:`StudentSearchResults` JSON file written in
        ``save_directory`` with the same name as the dataset. A question with
        no usable token gets an empty list of sources.

        Args:
            dataset_path: Path of a :class:`RagDataset` JSON file.
            k: Number of sources per question.
            save_directory: Directory where the results are saved.

        Raises:
            ValueError: If ``k`` is invalid or larger than the number of
                chunks.
            FileNotFoundError: If the dataset or the index does not exist.
            pydantic.ValidationError: If the dataset is malformed.
        """
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
        print(f"Saved to {dataset_path}")

    def answer(self, query: str, k: int = 5) -> None:
        """Retrieve the top-k sources and print the generated answer.

        Args:
            query: The question.
            k: Number of sources given to the model.

        Raises:
            ValueError: If ``k`` is invalid or the query has no usable
                token.
            FileNotFoundError: If the index does not exist.
        """
        self._check_k(k=k)
        retrivier = Retriever(index_dir=self.index_dir)
        sources = retrivier.search(query=query, k=k)
        lookup = ChunkLookUp(self.index_dir)
        llm_m = ManagerLLM()
        res = llm_m.generate(
            query=query,
            sources=sources,
            texts=[lookup.get(s) for s in sources]
        )
        print(res)

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> None:
        """Generate an answer for every question of a search result file.

        The output is a :class:`StudentSearchResultsAndAnswer` JSON file
        written in ``save_directory`` with the same name as the input file.

        Args:
            student_search_results_path: Path of a
                :class:`StudentSearchResults` JSON file.
            save_directory: Directory where the answers are saved.

        Raises:
            FileNotFoundError: If the input file or the index does not
                exist.
            pydantic.ValidationError: If the input file is malformed.
        """
        search_result = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
        )
        lookup = ChunkLookUp(self.index_dir)
        llm_m = ManagerLLM()
        answers: list[MinimalAnswer] = []

        for res in tqdm(search_result.search_results):
            texts = [lookup.get(s) for s in res.retrieved_sources]
            min_ans = MinimalAnswer(
                question_id=res.question_id,
                question=res.question,
                retrieved_sources=res.retrieved_sources,
                answer=llm_m.generate(
                    query=res.question,
                    sources=res.retrieved_sources,
                    texts=texts
                )
            )
            answers.append(min_ans)
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_result.k
        )
        (save_dir / Path(student_search_results_path).name).write_text(
            output.model_dump_json(indent=2)
        )
        print(f"Saved to {student_search_results_path}")

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str
    ) -> None:
        """Print the recall@k of search results against a ground truth.

        Recall@k is computed for every k in ``{1, 3, 5, 10}`` that is not
        larger than the ``k`` of the search results.

        Args:
            student_search_results_path: Path of a
                :class:`StudentSearchResults` JSON file.
            dataset_path: Path of an answered :class:`RagDataset` JSON file.

        Raises:
            ValueError: If no recall can be computed or the dataset has no
                answered question.
            FileNotFoundError: If a file does not exist.
            pydantic.ValidationError: If a file is malformed.
        """
        search_result = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text()
        )
        k_recall = [k for k in [1, 3, 5, 10] if k <= search_result.k]
        if not k_recall:
            raise ValueError("Something is wrong with the value k")
        tot_k = {
            k: 0.0
            for k in k_recall
        }
        data_set = RagDataset.model_validate_json(
            Path(dataset_path).read_text()
        )
        data_formated: list[AnsweredQuestion] = [
            data for data in data_set.rag_questions
            if isinstance(data, AnsweredQuestion) and len(data.sources) > 0
        ]
        if not data_formated:
            raise ValueError("Something is wrong with "
                             "the dataset_path in evaluate")
        search_dict = {
            search.question_id: search.retrieved_sources
            for search in search_result.search_results
        }

        for expected in data_formated:
            retrieve = search_dict.get(expected.question_id, [])
            for k in k_recall:
                tot_k[k] += self._recall_from_k(k, retrieve, expected.sources)
        for k in k_recall:
            print(f"Recall@{k}: {(tot_k[k] / len(data_formated)):.3f}")

    def _recall_from_k(self, k: int, retrieve: list[MinimalSource],
                       expected: list[MinimalSource]) -> float:
        """Compute the recall@k of one question.

        Args:
            k: Number of retrieved sources to consider.
            retrieve: Retrieved sources, best first.
            expected: Ground-truth sources.

        Returns:
            The fraction of expected sources found in the top-k results.
        """
        count: int = 0
        for source in expected:
            for r in retrieve[:k]:
                if r.file_path != source.file_path:
                    continue
                if self._is_overlap(r, source):
                    count += 1
                    break
        return count/len(expected)

    def _is_overlap(self, search: MinimalSource, source: MinimalSource
                    ) -> bool:
        """Tell whether a retrieved source covers an expected source.

        Args:
            search: Retrieved source.
            source: Expected source, in the same file.

        Returns:
            True if the ranges overlap enough.
        """
        search_span = (search.first_character_index,
                       search.last_character_index)
        source_span = (source.first_character_index,
                       source.last_character_index)
        overlap_span = (max(search_span[0], source_span[0]),
                        min(search_span[1], source_span[1]))
        overlap = overlap_span[1] - overlap_span[0]
        if overlap <= 0:
            return False
        return self._is_compute_overlap(overlap, source_span)

    def _is_compute_overlap(self, actual_overlap: int, span: tuple[int, int]
                            ) -> bool:
        """Tell whether an overlap covers at least 5% of a span.

        Args:
            actual_overlap: Number of overlapping characters.
            span: ``(first, last)`` range of the expected source.

        Returns:
            True if the overlap is at least 5% of the span length.
        """
        total_span = span[1] - span[0]
        required_overlap = int(total_span * 0.05)
        if actual_overlap >= required_overlap:
            return True
        return False
