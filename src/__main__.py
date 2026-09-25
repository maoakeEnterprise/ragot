from utils import Chunker, ChunkIndex, DataManager, Retriever


if __name__ == "__main__":

    scope = [
        'py',
        'md',
        'txt'
    ]
    index_dir = "data/processed"

    indexer = Chunker("data/raw/vllm-0.10.1/", scope)
    files = indexer.get_files()
    chunkIndex = ChunkIndex(chunks=indexer.chunk_files(list_path=files))

    data_m = DataManager()
    data_m.init_folder()
    data_m.register_chunk_file(
        data=chunkIndex.model_dump_json(),
        index_dir=index_dir)
    data_m.save_index_data(chunk_index=chunkIndex, index_dir=index_dir)
    retrivier = Retriever(
        index_dir=index_dir,
        tokenizer=data_m.get_tokenizer())
    ms_l = retrivier.search(query="How to use get_lora_path?", k=5)
    for ms in ms_l:
        print(f"FILE PATH: {ms.file_path}\n"
              f"first char id: {ms.first_character_index}\n"
              f"last char id: {ms.last_character_index}\n"
              )
