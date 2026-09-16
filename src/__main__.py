from utils import Indexer


if __name__ == "__main__":
    indexer = Indexer("data/raw/vllm-0.10.1/")
    test = indexer.get_files()
    for i in test:
        print(i)
