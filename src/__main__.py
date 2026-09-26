from utils import Rag
import fire


if __name__ == "__main__":
    fire.Fire(Rag)

    rag = Rag()
    rag.index()
    rag.search(query="How to use get_lora_path?", k=5)
