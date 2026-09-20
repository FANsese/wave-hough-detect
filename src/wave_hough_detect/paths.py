"""
数据文件定位

论文 §3.2 用的 MEA 记录不随本仓库分发（见 ``data/README.md``）。
本模块按优先级在若干常见位置查找它，让示例脚本不必硬编码路径。

查找顺序
--------
1. 显式传入的路径（命令行 ``--data``）
2. 环境变量 ``WHD_DATA``
3. 当前工作目录及其父目录下的 ``data/<文件名>`` 或直接 ``<文件名>``
4. 包所在目录往上三层的 ``data/<文件名>``

都找不到时抛出带说明的 ``FileNotFoundError``。
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_NAME = "CM_PIN_Control_2N30_Aunor-txt.csv"


def _candidates(name: str, explicit: str | Path | None) -> list[Path]:
    out: list[Path] = []
    if explicit is not None:
        out.append(Path(explicit))
    env = os.environ.get("WHD_DATA")
    if env:
        out.append(Path(env))

    here = Path.cwd().resolve()
    for base in [here, *here.parents[:2], Path(__file__).resolve().parents[3]]:
        out.append(base / "data" / name)
        out.append(base / name)
    return out


def find_recording(explicit: str | Path | None = None,
                   name: str = DEFAULT_NAME) -> Path:
    """
    定位 MEA 记录文件，返回其路径。

    参数
    ----
    explicit : 显式路径。给目录则在该目录下按 ``name`` 查找。
    name     : 数据文件名，默认是论文 §3.2 用的那份。

    异常
    ----
    FileNotFoundError : 所有候选位置都没有该文件。
    """
    for c in _candidates(name, explicit):
        p = c / name if c.is_dir() else c
        if p.is_file():
            return p

    tried = "\n".join(f"    {c}" for c in _candidates(name, explicit))
    raise FileNotFoundError(
        f"找不到数据文件 {name}\n"
        f"已尝试以下位置：\n{tried}\n\n"
        "该记录是合作实验室的实验数据，不随仓库分发（见 data/README.md）。\n"
        "可用三种方式指定：\n"
        "  1. 命令行  --data /path/to/recording.csv\n"
        "  2. 环境变量  export WHD_DATA=/path/to/recording.csv\n"
        "  3. 放到仓库的 data/ 目录下\n\n"
        "若只是想试跑管线，不需要任何实验数据：\n"
        "    python examples/demo_simulation.py\n"
        "它会用 simulate_recording() 现生成一份同格式的 64 通道合成记录并跑完整管线，\n"
        "而且结果会与它自己设定的真值逐项对照。"
    )
