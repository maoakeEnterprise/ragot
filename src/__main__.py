"""Command-line entry point of the RAG system.

Exposes the :class:`utils.Rag` commands through Python Fire and turns every
expected exception into a readable error message instead of a traceback.

Example:
    uv run python -m src search "How to configure the OpenAI server?" --k 5
"""
from utils import Rag
import fire
from pydantic import ValidationError
import sys
import torch


def print_error(header: str, message: str, e: BaseException) -> None:
    """Print a formatted error on stderr and exit with status 1.

    Args:
        header: Banner line describing the kind of error.
        message: Short human-readable explanation.
        e: The exception that was caught.
    """
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
    except torch.OutOfMemoryError as e:
        print_error("=============OUT=OF=MEMORY=ERROR============",
                    "NOT ENOUGH MEMORY TO LOAD OR RUN QWEN", e)
    except OSError as e:
        print_error("=============OS=ERROR=======================",
                    "SOMETHING GOT WRONG (MODEL DOWNLOAD / FILE)", e)
