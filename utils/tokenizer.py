from utils.text_utils import STOP_WORDS
import re


class Tokenizer:
    def __init__(self) -> None:
        self.stop_words: set[str] = STOP_WORDS
        self.word_re = re.compile("[A-Za-z_][A-Za-z0-9_]*")
        self.camel_re = re.compile(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
            )

    def tokenize(self, text: str) -> list[str]:
        pass

    def _extract_words(self, text: str) -> list[str]:
        pass

    def _split_identifier(self, word: str) -> list[str]:
        pass

    def _is_kept(self, text: str) -> str:
        pass
