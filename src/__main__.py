from utils import Chunker, Tokenizer
import bm25s


if __name__ == "__main__":

    scope = [
        'py',
        'md',
        'txt'
    ]

    indexer = Chunker("data/raw/vllm-0.10.1/", scope)
    files = indexer.get_files()
    chunked = indexer.chunk_files(list_path=files)
    tokenizer = Tokenizer()
    bm25 = bm25s.BM25(k1=1.5, b=0.75)
    token_list = [tokenizer.tokenize(chunk.content) for chunk in chunked]
    bm25.index(token_list)
    question_token = tokenizer.tokenize("How to configure LoRA?")
    test = bm25.retrieve(query_tokens=[question_token], k=5)
    for i in range(len(test.documents[0])):
        print(chunked[test.documents[0][i]].file_path, test.scores[0][i])
