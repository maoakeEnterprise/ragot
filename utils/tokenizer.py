from utils.text_utils import STOP_WORDS


class Tokenizer:
    def __init__(self) -> None:
        self.stop_words: set[str] = STOP_WORDS
        self.word_reg = "[A-Za-z_][A-Za-z0-9_]*"
