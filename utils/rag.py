from utils import Chunker, ChunkIndex, DataManager, Retriever


class Rag:
    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str = "data/raw/",
            output_dir: str = "data/processed/"
    ) -> None:
        scope = [
                'py',
                'md',
                'txt'
            ]
        data_m = DataManager()
        indexer = Chunker(path=f"{raw_dir}vllm-0.10.1/", scope=scope)
        files = indexer.get_files()
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

    def searh(self, query: str, k: int = 5) -> None:
        retriever = Retriever()
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
        pass

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
