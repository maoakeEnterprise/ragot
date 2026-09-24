from utils.text_utils import STOP_WORDS
import re


class Tokenizer:
    def __init__(self) -> None:
        self.stop_words: set[str] = STOP_WORDS.copy()
        self.word_re = re.compile("[A-Za-z_][A-Za-z0-9_]*")
        self.camel_re = re.compile(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
            )

    def tokenize(self, text: str) -> list[str]:
        words = self._extract_words(text)
        res: list[str] = []
        for word in words:
            if self._is_kept(word):
                tmp = self._split_identifier(word)
                if len(tmp) > 1:
                    res.append(word)
                tmp = [p for p in tmp if self._is_kept(p)]
                res.extend(tmp)
        final_res = [p.lower() for p in res]
        if len(final_res) == 0:
            raise ValueError("Something is wrong")
        return final_res

    def _extract_words(self, text: str) -> list[str]:
        return self.word_re.findall(text)

    def _split_identifier(self, word: str) -> list[str]:
        tab = word.split("_")
        res: list[str] = []
        for x in tab:
            res.extend(self.camel_re.split(x))
        return [i.lower() for i in res if i != '']

    def _is_kept(self, text: str) -> bool:
        if len(text) < 2:
            return False
        if text.lower() in self.stop_words:
            return False
        return True
