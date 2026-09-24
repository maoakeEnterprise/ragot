from utils.text_utils import STOP_WORDS
import re


class Tokenizer:
    def __init__(self) -> None:
        self.stop_words: set[str] = STOP_WORDS.copy()
        self.stop_words.add("how")
        self.word_re = re.compile("[A-Za-z_][A-Za-z0-9_]*")
        self.camel_re = re.compile(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
            )

    def tokenize(self, text: str) -> list[str]:
        words = self._extract_words(text)
        res: set[str] = set()
        for word in words:
            if self._is_kept(word):
                res.update(self._split_identifier(word))
                res.add(word)
        return [p.lower() for p in res]

    def _extract_words(self, text: str) -> list[str]:
        return self.word_re.findall(text)

    def _split_identifier(self, word: str) -> list[str]:
        tab = word.split("_")
        res: list[str] = []
        for x in tab:
            res.extend(self.camel_re.split(x))
        return [i.lower() for i in res]

    def _is_kept(self, text: str) -> bool:
        if len(text) < 2:
            return False
        if text.lower() in self.stop_words:
            return False
        return True
