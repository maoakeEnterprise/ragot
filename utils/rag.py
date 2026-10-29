from utils import (Chunker, ChunkIndex, DataManager, Retriever, RagDataset,
                   MinimalSearchResults, StudentSearchResults,
                   ManagerLLM, ChunkLookUp, MinimalAnswer,
                   StudentSearchResultsAndAnswer, AnsweredQuestion,
                   MinimalSource)
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
        print(f" Ingestion complete! Indexed {len(chunk_index.chunks)} chunks "
              f"under {output_dir}")

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
        print(f"Saved to {dataset_path}")

    def answer(self, query: str, k: int = 5) -> None:
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
        total_span = span[1] - span[0]
        required_overlap = int(total_span * 0.05)
        if actual_overlap >= required_overlap:
            return True
        return False
