from utils import Indexer


if __name__ == "__main__":

    scope = {
        'py',
        'md',
        'txt'
    }

    indexer = Indexer("data/raw/vllm-0.10.1/", scope)
    test = indexer.get_files()
    for i in test:
        print(i)
