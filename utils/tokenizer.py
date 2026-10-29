"""Code-aware tokenizer for BM25."""
from utils.text_utils import STOP_WORDS
import re
from tqdm import tqdm


class Tokenizer:
    """Split text and code into lowercase BM25 tokens.

    Identifiers are kept whole and also split on ``snake_case`` and
    ``CamelCase``, so a query can match both a verbatim identifier and its
    parts.

    Attributes:
        stop_words: English stop words to remove.
        word_re: Regex matching identifiers and words.
        camel_re: Regex matching ``CamelCase`` boundaries.
    """

    def __init__(self) -> None:
        """Initialize the stop words and the regexes."""
        self.stop_words: set[str] = STOP_WORDS.copy()
        self.word_re = re.compile("[A-Za-z_][A-Za-z0-9_]*")
        self.camel_re = re.compile(
            r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])"
            )

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a text.

        Example:
            ``"gpu_memory_utilization"`` gives
            ``["gpu_memory_utilization", "gpu", "memory", "utilization"]``.

        Args:
            text: Text or source code to tokenize.

        Returns:
            Lowercase tokens, without stop words and tokens shorter than
            2 characters.
        """
        words = self._extract_words(text)
        res: list[str] = []
        for word in tqdm(words):
            if self._is_kept(word):
                tmp = self._split_identifier(word)
                if len(tmp) > 1:
                    res.append(word)
                tmp = [p for p in tmp if self._is_kept(p)]
                res.extend(tmp)
        final_res = [p.lower() for p in res]
        return final_res

    def _extract_words(self, text: str) -> list[str]:
        """Extract every word and identifier of a text.

        Args:
            text: Text to scan.

        Returns:
            The matches of ``word_re``.
        """
        return self.word_re.findall(text)

    def _split_identifier(self, word: str) -> list[str]:
        """Split an identifier on underscores and ``CamelCase``.

        Args:
            word: Identifier to split.

        Returns:
            The lowercase parts of the identifier.
        """
        tab = word.split("_")
        res: list[str] = []
        for x in tab:
            res.extend(self.camel_re.split(x))
        return [i.lower() for i in res if i != '']

    def _is_kept(self, text: str) -> bool:
        """Tell whether a token is kept.

        Args:
            text: Token to check.

        Returns:
            False for stop words and tokens shorter than 2 characters.
        """
        if len(text) < 2:
            return False
        if text.lower() in self.stop_words:
            return False
        return True
