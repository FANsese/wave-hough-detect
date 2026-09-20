"""
pytest 配置。

仓库用的是 src/ 布局，测试直接跑在源码树上（无需先 ``pip install -e .``），
所以这里把 src/ 加进 import 路径。
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
