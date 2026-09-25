import sys
from pathlib import Path

import pytest
import torch

# The benchmarks are standalone scripts; make them importable in tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

torch.set_num_threads(4)


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")
