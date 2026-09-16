from pathlib import Path


class Indexer:

    def __init__(self, path: str) -> None:
        self.path: Path = Path(path)

    def get_files(self, path: str | None = None) -> None:
        list_path: list[Path] = []
        if path is not None:
            path_p = Path(path)
            list_path = self._algo_files(path_p)
        else:
            list_path = self._algo_files(self.path)
        return list_path

    def _algo_files(self, path: Path) -> list[Path]:
        list_path: list[Path] = []
        for path_file in path.rglob("*"):
            if path_file.is_file():
                list_path.append(path_file)
        return list_path
