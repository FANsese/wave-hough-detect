"""
dev 验证脚本共用的路径与导入引导。

这一目录【只在本机使用】，不属于发布内容：它需要论文工程根目录下的
真实数据与 R 侧产物作为比对基准，这些都不随仓库分发。

目录关系
--------
    <论文工程根目录>/                      ← PROJ
    ├── CM_PIN_Control_2N30_Aunor-txt.csv  ← DATA（真实记录，不可分发）
    ├── R_work/output/                     ← R_OUT（R 侧基准产物）
    └── wave-hough-detect-py/              ← REPO
        └── dev/
            ├── out/                       ← OUT（Python 侧产物，可随时删）
            └── verify_against_r/          ← HERE（本目录）
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                  # wave-hough-detect-py/
PROJ = REPO.parent                      # 论文工程根目录
OUT = REPO / "dev" / "out"              # Python 侧产物
R_OUT = PROJ / "R_work" / "output"      # R 侧产物
DATA = PROJ / "CM_PIN_Control_2N30_Aunor-txt.csv"

# 允许不安装直接运行：把仓库的 src/ 加入搜索路径。
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

# 基准产物是 R 生成的，缺了就无从比对，提前给出可操作的提示。
R_REQUIRED = {
    "step2_hough.py": ["trace.csv", "hough_result.csv", "hough_planes.csv"],
    "step3_fit.py": ["trace.csv", "result_array_r.csv",
                     "fit_circular_r.csv", "fit_linear_r.csv"],
}


def hr(title: str = "") -> None:
    print()
    print("=" * 74)
    if title:
        print(f" {title}")
        print("=" * 74)


def require_r_outputs(script_name: str) -> bool:
    """检查 R 侧基准产物是否齐备；缺哪个就告诉用户去跑哪个脚本。"""
    missing = [f for f in R_REQUIRED.get(script_name, [])
               if not (R_OUT / f).exists()]
    if not missing:
        return True
    print(f"\n❌ 缺少 R 侧基准产物：{', '.join(missing)}")
    print("   它们由 R 脚本生成，请先在论文工程根目录执行：")
    if "trace.csv" in missing:
        print("     Rscript R_work/run2_trace.R   # → trace.csv, "
              "hough_result.csv, hough_planes.csv")
    if any(f.endswith("_r.csv") for f in missing):
        print("     Rscript R_work/run3_fit.R     # → result_array_r.csv, "
              "fit_circular_r.csv, fit_linear_r.csv")
    print(f"   R 产物目录：{R_OUT}\n")
    return False
