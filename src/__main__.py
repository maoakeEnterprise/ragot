from utils import Chunker


if __name__ == "__main__":

    scope = [
        'py',
        'md',
        'txt'
    ]

    indexer = Chunker("data/raw/vllm-0.10.1/", scope)
    files = indexer.get_files()
    chunked = indexer.chunk_files(list_path=files)
    for chunk in chunked:
        print(f"FIRST CHAR: {chunk.first_character_index}")
        print(f"LAST CHAR: {chunk.last_character_index}")
