from utils import Rag
import fire
from pydantic import ValidationError
import sys


def print_error(header: str, message: str, e: BaseException) -> None:
    print(f"{header}\n"
          f"{message}\n"
          f"ERROR: {e}\n"
          "============================================", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    try:
        fire.Fire(Rag)
    except ValueError as e:
        print_error("=============VALUE=ERROR====================",
                    "SOMETHING GOT WRONG", e)
    except FileNotFoundError as e:
        print_error("=============FILE=NOT=FOUND=ERROR===========",
                    "SOMETHING GOT WRONG", e)
    except IsADirectoryError as e:
        print_error("=============IS=A=DIRECTORY=ERROR===========",
                    "SOMETHING GOT WRONG NICE TRY BROTHER", e)
    except PermissionError as e:
        print_error("=============PERMISSION=ERROR===============",
                    "SOMETHING GOT WRONG", e)
    except ValidationError as e:
        print_error("=============VALIDATION=ERROR===============",
                    "SOMETHING GOT WRONG", e)
    except TypeError as e:
        print_error("=============TYPE=ERROR=====================",
                    "SOMETHING GOT WRONG NICE TRY BROTHER", e)
    except KeyboardInterrupt as e:
        print_error("=============KEYBOARD=INTERRUPT=ERROR=======",
                    "SOMETHING GOT WRONG", e)
