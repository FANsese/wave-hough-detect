"""
dev 验证脚本共用的路径与导入引导。

【这一目录是什么】
用来证明「Python 版与原始 R 实现逐步算出同样的结果」。这不是常规单元测试，
而是移植的验收证据：判据是【逐位相同】，不是「差不多」。

【为什么不在 tests/ 里】
它需要两样不随仓库分发的东西：
  · 论文 §3.2 的实验记录（合作实验室产出，见 data/README.md）
  · R 侧的基准产物（由 R 脚本生成，见 docs/porting-and-validation.md §5）
干净 clone 里这两样都没有，所以它单独放在 dev/ 下，按需手动运行。
不需要数据的部分全在 tests/ 里，`pytest -q` 就能跑。

目录关系
--------
    <论文工程根目录>/                      ← PROJ
    ├── CM_PIN_Control_2N30_Aunor-txt.csv  ← DATA（实验记录，不随仓库分发）
    ├── R_work/output/                     ← R_OUT（R 侧基准产物）
    └── wave-hough-detect/                 ← REPO
        └── dev/
            ├── out/                       ← OUT（Python 侧产物，可随时删）
            └── verify_against_r/          ← HERE（本目录）
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                  # wave-hough-detect/
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
