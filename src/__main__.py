from utils import Chunker, Tokenizer, ChunkIndex, DataManager
import bm25s


if __name__ == "__main__":

    scope = [
        'py',
        'md',
        'txt'
    ]

    indexer = Chunker("data/raw/vllm-0.10.1/", scope)
    files = indexer.get_files()
    chunkIndex = ChunkIndex(chunks=indexer.chunk_files(list_path=files))

    DataManager.init_folder()
    DataManager.create_chunk_file(chunkIndex.model_dump_json())

    tokenizer = Tokenizer()
    bm25 = bm25s.BM25(k1=1.5, b=0.75)

    bm25.index([
        tokenizer.tokenize(chunk.content)
        for chunk in chunkIndex.chunks
    ])

    bm25.save("data/processed/bm25")

    bm25 = DataManager.load_index_file()
    chunkIndex = DataManager.load_chunk_file()
    question_token = tokenizer.tokenize("How to configure LoRA?")
    test = bm25.retrieve(query_tokens=[question_token], k=5)
    print(test)
