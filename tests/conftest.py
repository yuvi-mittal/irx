"""General configuration module for pytest."""

import os
import tempfile

from difflib import SequenceMatcher
from pathlib import Path

import astx

from irx.builders.base import Builder

TEST_DATA_PATH = Path(__file__).parent / "data"


def similarity(text_a: str, text_b: str) -> float:
    """Calculate the similarity between two strings."""
    return SequenceMatcher(None, text_a, text_b).ratio()


def check_result(
    action: str,
    builder: Builder,
    module: astx.Module,
    expected_file: str = "",
    similarity_factor: float = 0.35,  # TODO: change it to 0.95
) -> None:
    """Check the result for translation or build."""
    if action == "build":
        filename_exe = ""
        with tempfile.NamedTemporaryFile(
            suffix=".exe",
            prefix="arx",
            dir="/tmp",
            delete=False,
        ) as fp:
            filename_exe = fp.name
            builder.build(module, output_file=filename_exe)
        builder.run()
        os.unlink(filename_exe)
    elif action == "translate":
        with open(TEST_DATA_PATH / expected_file, "r") as f:
            expected = f.read()
        result = builder.translate(module)
        print(" TEST ".center(80, "="))
        print("==== EXPECTED =====")
        print(f"\n{expected}\n")
        print("==== results =====")
        print(f"\n{result}\n")
        print("=" * 80)
        assert similarity(result, expected) >= similarity_factor
