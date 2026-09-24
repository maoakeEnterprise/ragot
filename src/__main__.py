from utils import Chunker, Tokenizer


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
    print(tokenizer.tokenize("LoRARequest"))
    print(tokenizer.tokenize("How to use get_lora_path?"))
