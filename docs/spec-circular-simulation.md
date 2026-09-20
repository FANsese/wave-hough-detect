# 圆波前仿真 + 交替最小化拟合：Python 移植实现规范

> **状态：已实现；本文保留为逐行依据与差异记录。**
> 论文 §3.1.2（96×96 圆波前仿真，图 6–8）的 Python 实现已完成，见
> `../src/wave_hough_detect/simulate_circular.py` 与 `../examples/demo_simulation_circular.py`。
>
> 本文最初是**照着 R 源码整理出来的实现规范**，现在有四个用处：
>   1. 逐行记录 §3.1.2 的参数、抽样结构与评估口径（论文里没有全部写出来）；
>   2. 记录 R 代码里那些"名不副实"的地方（例如 `relerror.plot` 画的是绝对值而不是相对误差）；
>   3. **记录哪些部分没有被移植、以及为什么** —— 见 `docs/code-paper-mapping.md` 的
>      "Not ported from this file" 一段；
>   4. 给出成本量级与不确定项，供你与论文原文核对。
>
> 文中标 **⚠️ 不确定** 的地方是静态阅读无法确证的，**不要当作结论使用**；
> 标 **✅ 已验证** 的地方做过等价的纯 Python 数值验证。
>
> 实现完成之后又实测确认/新增的三条（细节见 `docs/reproducibility-notes.md`）：
>   · R 的初值 (u, t₀) = (1, 1) 在 §3.1.2 **恰好几乎正中真值**（v = 1、t₀ = 2），
>     所以 §3.1.1 那个"源点在网格内会收敛错"的毛病在这里不会发作；
>   · `r2` 是对**信号 + 噪声全部点**算的，所以模型精确时也只有 ≈0.953，
>     不能与论文表 4 的 0.94 并列比较；
>   · σ 扫描（mode 3）被噪声尖峰淹没：λ_n = 1 时那 ≈100 个离群点主导损失，
>     σ 从 0.1 到 1.0 相对误差变化不到 2 倍，把噪声关掉才涨 10 倍以上。

> 目标源文件：`R_Codes/stomach_hough17_newton_square20.r`（1012 行，Jul 2025 版本）
> 参照副本：`R_Codes/stomach_plane_anglediff14.r`（同目录，**内容不同**，见 §0.3）
> 已发布包内的同名副本：`whd-20260915release/R/02_simulation/fig7_fig8_fitting_accuracy.R`
> —— 经 `diff` 校验，与本文件**逐字节相同**（1012 行），故本文所有行号对二者都成立。
>
> 本文所有引用格式为 `stomach_hough17_newton_square20.r:123`。
> 标注 **⚠️ 不确定** 的地方是我无法仅凭静态阅读确证的（本机无 R 环境），**不要**当作结论使用。
> 标注 **✅ 已验证** 的地方是我用等价的纯 Python 实现做过数值验证的。

---

## 0. 文件的执行模型（先读这一节）

### 0.1 顶层依赖与副作用

| 行 | 语句 | 作用 / 移植影响 |
|---|---|---|
| `:1` | `options(error=recover)` | 出错时进入交互式调试器。批处理/Rscript 下会尝试从 stdin 读取选择，**可能挂起**。Python 端忽略。 |
| `:2` | `require(hash)` | 只被 `hough()`（`:747`）使用，而 `hough` 在本文件的活跃路径上**不被调用**。 |
| `:3` | `require(rgl)` | 见 §5.2；**加载期**硬依赖，headless 机器上是第一道风险。 |
| `:4` | `require(numDeriv)` | 只被 `newton.cone.square()`（`:539`,`:542`）使用，该函数在活跃路径上不被调用。 |
| `:11-13` | `dir.create("plots")` | 输出 PDF/PNG 目录。 |
| `:371-373` | `LIMIT = 96; x.grid = array(1:LIMIT, c(LIMIT,LIMIT)); y.grid = t(x.grid)` | **全局**网格常量。`norm.xy/dx0/.../hes/cone.model/cone.model.square/cone.model.inverted` 全部**隐式**引用它们（§6 陷阱 T21）。 |
| `:1010` | `optimality.table()` | **顶层调用**：`source()` 或 `Rscript` 本文件即执行整轮扫描。 |

`require()` 与 `library()` 的区别：缺包时 `require` 只返回 `FALSE` 并打印警告，**不中断**，脚本会继续跑到第一次真正用到该包的函数时才报错。移植时不要把它当成硬性依赖检查。

### 0.2 函数可达性（哪些是活代码，哪些是历史遗留）

这是本文件最容易误判的地方：文件里定义了近 40 个函数，但**只有 8 个**参与论文图 6/7/8 的产出。

| 状态 | 函数 | 定义行 |
|---|---|---|
| ✅ **活跃**（图 6/7/8 的实际计算路径） | `optimality.table` → `visualize2d` → `stomach.sim2d` → `stomach.sim.one2d`；`hybrid.optim` → `grid.optim` → `grid.optim.find.best.point` → `grid.optim.find.best.point.1d` → `cone.model.inverted.all.xy` → `cone.model.inverted.all`；`vt.optim.all`；`normv`（`:531`，被 `grid.optim:603` 与 `hybrid.optim:645` 调用）；`relerror.plot` | `:878` `:652` `:156` `:100` `:631` `:586` `:579` `:556` `:337` `:316` `:620` `:782` |
| ☠️ 死代码（定义完整、从未被活跃路径调用） | `stomach.sim`(`:60`)、`stomach.sim.one`(`:32`)、`cone.model`(`:290`)、`cone.model.inverted`(`:303`)、`cone.model.inverted.xy`(`:332`)、`cone.model.square`(`:342`)、`convert.to.grid`(`:357`)、`norm.xy`(`:375`)、`dx0/dy0/dv/dt0/dx0x0/dy0y0/dvv/dt0t0/dx0y0/dx0v/dx0t0/dy0v/dy0t0/dvt0`(`:380-491`)、`gra`(`:493`)、`hes`(`:502`)、`newton.cone`(`:517`)、`newton.cone.square`(`:533`)、`vt.optim`(`:608`)、`compute.theta`(`:200`)、`visualize`(`:244`)、`hough`(`:745`)、`point.angle.to.line`(`:732`) | 见左 |

被注释掉、**不要**实现的部分：`cone.model.inverted.vt`（`:327-330`）、`visualize` 里的 3D 分块（`:688-726`）、`visualize2d` 里 `ts.grid` 的调用（`:670`）、`optimality.table` 末尾的旧扫描循环（`:1000-1007`）。

**结论**：论文图 6/7/8 的整条链路里**完全没有 Hough 变换**。`hough()`/`point.angle.to.line()`/`visualize()` 是 1D `stomach_hough*.r` 时代留下的遗留代码（其语法与 2D 模拟器的数据布局也不自洽：`:256` 的 `hough(ts, x)`），在图 6/7/8 的复现里只需按 §4.2 记录行为，不必实现。**移植优先级：低。**

### 0.3 与 `stomach_plane_anglediff14.r` 的关系

两个文件共享**约 60%** 的同名函数（`stomach.sim*`、`hough`、`relerror.plot` 等），但差异是实质性的，**不要**把二者当作同一份代码：

| 差异点 | 本文件（primary） | `stomach_plane_anglediff14.r` |
|---|---|---|
| `stomach.sim.one2d` | **逐格遍历**（`:109-139`），`p` 缺失机制**生效** | `while` + `stomach.sim.one`（`:99-108`），**2D 圆波前完全缺失** |
| `stomach.sim2d` 的 `SEED_BASE` | `100`（`:158`） | `101`（`:120`），且有 `seed.shift` 形参 |
| 波间隔 | `rexp(1, 1/gap)`（`:171`） | `rchisq(1, gap, ncp=0)`（`:139`） |
| 种子表达式 | `...*miux*miuy*sigma`（`:160-161`），**无** `seed.shift` | `...*miu*sigma + seed.shift`（`:128-129`） |
| 建模部分 | `cone.model*` + 解析 `gra/hes` + `newton.cone.square` | 平面模型 `get.ts/draw.plane/plane.model` |
| `relerror.plot` | 同名同义 | 同名同义（另一份拷贝） |

**⚠️ 已发布文档 `whd-20260915release/docs/reproducibility-notes.md:36-38` 把种子表达式写成 `SEED_BASE * ... * miu * sigma + seed.shift`（`SEED_BASE = 1000000`）——那描述的是 `fig5_detection_performance.R` 里的 `stomach.sim2d`，对本文件不成立。** 本文件的真实表达式见 §1.2，其中没有 `seed.shift`。

---

## 1. 数据模拟器 / The data simulator

### 1.1 真实函数名与签名

模拟 96×96 圆波前数据集的函数是 **`stomach.sim2d`**（定义 `:156-198`），不是 `stomach.sim3d`；内部逐格生成由 **`stomach.sim.one2d`**（`:100-153`）完成。

```r
# :156-157
stomach.sim2d = function(limit, time.max, ts, x, y, u.x, u.y, p, noise.freq,
                         miux, miuy, sigma, gap, ratio)
# :100-101
stomach.sim.one2d = function(limit, time.max, ts, x, y, u.x, u.y, p,
                             noise.freq, miux, miuy, sigma, ratio)
```

唯一调用点 `stomach_hough17_newton_square20.r:661-662`（在 `visualize2d` 内），**逐参数对应**如下。
`visualize2d` 的局部常量定义在 `:653-659`：

| 形参 | 调用点实参 | 值（本文件配置） | 含义 |
|---|---|---|---|
| `limit` | `limit` | **96**（`:653`） | 网格边长；坐标范围 1…96 |
| `time.max` | `time.max` | **96**（`:656`，`time.max = limit`） | 时间上限 |
| `ts` | `ts` | **2**（`:659`） | 激发时刻 t₀ |
| `x`,`y` | `x`,`y` | **48, 48**（`:659`） | 波源中心 (x₀,y₀) |
| `u.x`,`u.y` | `u.x`,`u.y` | **1, 1**（`:659`） | **在本模拟器中完全未使用**（见 T34） |
| `p` | `p` | `theta$p`（`:658`） | 逐电极**缺失概率** |
| `noise.freq` | `lambda` | `theta$lambda`（`:658`） | **噪声 Poisson 率**（每单位时间一个噪声点的强度），不是"SNR" |
| `miux`,`miuy` | `miux`,`miuy` | **1, 1**（`:659`，`miuy = miux`） | 各轴波速（各向同性时相等） |
| `sigma` | `sigma` | `theta$sigma`（`:658`） | 到达时刻的**测量噪声标准差** |
| `gap` | `gap` | **1000000**（`:657`） | 相邻两次激发的平均间隔 |
| `ratio` | `ratio` | **10**（`:659`） | **在本模拟器中完全未使用**（见 T34） |

三个"生成参数"由 `optimality.table` 的每个扫描点构造（`theta.true`），例如 `:928`：`list("lambda"=0.000001, "p"=0, "sigma"=sigma)`；其余几何/动力学参数一律硬编码在 `:653-659`。**注意命名陷阱**：`theta$lambda` 在 `visualize2d` 里被绑定到形参 `noise.freq`（第 9 个位置参数），它不是"波的发生率"，也与 `gap` 无关。

### 1.2 随机种子：如何由参数派生

```r
# :158-161
SEED_BASE = 100;
set.seed(SEED_BASE * limit * time.max * ts * x * y * u.x * u.y * p * noise.freq *
           miux * miuy * sigma)
```

代入调用点的值，种子 = **`100 · 96 · 96 · 2 · 48 · 48 · p · noise.freq · 1 · 1 · sigma`**，其中

```
100 · 96 · 96 · 2 · 48 · 48 = 4_246_732_800
```

即 `seed = 4.2467328e9 × p × noise.freq × sigma`（`u.x=u.y=miux=miuy=1` 只是乘 1）。

由此推出三条**必须写进移植实现**的结论（也是本文件最反直觉的地方）：

1. **`p = 0` 时种子恒为 0。** 乘积中出现因子 `p`，所以 `plot.mode ∈ {1,3,4,8}`（都令 `p=0`）的**每一个**扫描点都执行 `set.seed(0)`。
   ⇒ **当前提交的运行模式（`plot.mode = 3`，`:880`）的 10 个"重复实验"共享同一条 RNG 流**：9216 次 `rnorm` 抽出的标准正态序列完全相同，只是被乘上了不同的 `sigma`；噪声点数为 0。它们**不是统计独立的重复实验**。
   ⚠️ 这与 `whd-20260915release/docs/reproducibility-notes.md:55-59` 的说法（"独立性来自种子表达式中含被扫描参数"）直接冲突——该说法对 mode 2/5/7 成立，对 mode 3 不成立。**⚠️ 不确定**：论文图 8 究竟由 mode 2（`p` 扫描，种子随 `p` 变化，独立）还是 mode 3 产出；若是 mode 3，则图中的"误差条/离散度"语义与独立重复实验不同。
2. **种子被隐式取整。** `set.seed()` 需要整数；传入浮点时 R 会做 `as.integer` 式强制转换（**向零截断**）。例如 mode 2 的 `p = 0.1` 给出 `seed = 424.67328 → 424`；mode 6（`:965`，`lambda = 1e-7`）给出 `seed = 4.2467e-4 · p → 0`，于是 mode 6 的所有扫描点**又**共享 `set.seed(0)`。**⚠️ 不确定**：截断 vs 四舍五入我无法在本机验证（无 R 环境），但它直接影响复现"哪一次实验和哪一次相同"。Python 端若要严格模仿，应显式写 `int(seed)`（向零截断）并在文档里注明。
3. **存在整数溢出悬崖。** `as.integer(>2^31-1)` 给出 `NA` 并告警，`set.seed(NA)` 直接报错 `supplied seed is not a valid integer`。阈值：`p·noise.freq·sigma < 0.5057`。本文件所有已配置的 mode 都安全（mode 4 是 `p=0`；mode 2 是 `1e-6·p`），但**只要把 `sigma` 调到 0.5 以上并同时给 `p ≈ 0.8`、`lambda ≈ 1`，就会崩**（`4.2467e9 × 0.8 × 1 × 1 = 3.4e9 > 2^31`）。移植时建议 `seed = int(abs(...)) % 2**31` 之类显式包裹，并在文档里声明这是**有意的偏离**。

### 1.3 `stomach.sim2d` 的精确控制流与 RNG 调用顺序

RNG 调用顺序（**这是"复现随机结构"的全部内容**；顺序错一个就全错）：

```
1. set.seed(4.2467328e9 · p · noise.freq · sigma)                :160-161
2. signal.start.ts = ts = 2                                      :163
3. while (signal.start.ts < time.max) {                          :165
3a.   stomach.sim.one2d(...)                                     :166-167  → 见 3a.i–3a.iii
3b.   signal.start.ts += rexp(1, rate = 1/gap) = rexp(1, 1e-6)   :171
   }
4. noise.ts = 0                                                  :176
5. while (last(noise.ts) < time.max)                             :177
     noise.ts = append(noise.ts, last(noise.ts) + rexp(1, noise.freq))   :178
6. 裁剪：若 length(noise.ts) > 2 → 去掉首元素 0 与末元素(>time.max)      :180-184
7. noise.x = sample(1:limit, length(noise.ts), replace = TRUE)   :186
8. noise.y = sample(1:limit, length(noise.ts), replace = TRUE)   :187
9. 拼接 data.frame(x, y, ts, z) 并按 ts 升序排序                  :189-195
```

其中 3a（`stomach.sim.one2d`，`:100-153`）内部是一个**双重循环，i 外层、j 内层**（`:109-110`），共 `limit² = 9216` 次：

```
3a.i   for i in 1:limit { for j in 1:limit {
         distance = (i-x)^2 + (j-y)^2                                   :121
         e = rnorm(1, mean = 0, sd = sigma)          ← 每格必调用一次     :123
         if (distance == 0) arrival.time = ts + e                       :125-126
         else arrival.time = ts + sqrt(((i-x)/miux)^2 + ((j-y)/miuy)^2) + e   :128
         if (arrival.time < time.max) {                                 :131
           if (runif(1, 0, 1) > p) {    ← 条件调用，仅"及时到达"的格子     :132
             记录 (i, j, arrival.time)                                  :133-135
           }
         }
       }}
```

要点：
* `rnorm` **无条件**对全部 9216 个格子调用（即使该格随后被 `arrival.time` 或 `p` 过滤掉）；`runif` **有条件**调用。移植时必须保持这个顺序，否则 RNG 流错位。
* 判断用严格小于 `arrival.time < time.max`（`:131`）。
* **在本文件的配置下这个过滤永远为真**：`ts = 2`，最大距离 `sqrt(47²+47²) = 66.47`，故 `arrival.time ≤ 68.5 + e < 96`（`sigma ≤ 1` 时依然成立）。⇒ 9216 个格子全部进入"是否缺失"的判定。这一点决定了 §4.4 的 SNR 数值。
* `p = 0` 时 `runif(1,0,1) > 0` 几乎必然为真（`runif` 恰好返回 0 的概率为 0），故 mode 1/3/4/8 的信号点数**恰好 9216**。
* **"单波"是一个概率性断言，不是硬保证。** `gap = 1e6` 使第一次 `rexp(1, 1e-6)` 的均值约 1e6，`ts = 2` 时循环几乎必然只跑一次；但增量小于 `time.max − ts = 94` 的概率是 `1 − e^{−94·10⁻⁶} ≈ 9.4×10⁻⁵`，此时会再生成**第二个**波（其部分到达时刻会被 `:131` 过滤掉）。由于该 `rexp` 出现在第 `9216` 个 `rnorm` + 至多 `9216` 个 `runif` 之后，**在同一模式下 10 个重复实验的这个事件是同步的**（要么全有第二个波、要么全没有）。移植时若改变抽样顺序，可能导致某些重复实验多/少一个波，行数随之变化（T12b）。
* `within.range()`（`:103-105`）在这个函数里**从未被调用**（用到它的循环在 `:141-150` 被注释掉），而且它引用的是全局 `LIMIT`（`:371`）而非形参 `limit` —— 潜在 bug，但只是死代码（T13）。

### 1.4 噪声点的生成

* `:177-178` 用**速率参数** `noise.freq` 的指数间隔累加，直到超过 `time.max`。需要的增量个数 `K ≈ time.max · noise.freq + 1`，裁剪后保留 `K - 1 ≈ time.max · noise.freq` 个噪声点。
  ⇒ **`noise.freq` 就是"每单位时间的噪声点数"**（Poisson 强度），本文件中 `time.max = 96`。
* 噪声位置 `noise.x`、`noise.y` 是 1…96 上的**均匀独立**抽样（`:186-187`，`replace=TRUE`）。
* `:180-184` 的裁剪逻辑：`length(noise.ts) > 2` 时去掉首尾；否则（即恰好 1 次增量就超过 `time.max`）`noise.ts = c()`，**0 个噪声点**。这正是 mode 3（`lambda = 1e-6`，一次指数增量均值 1e6 ≫ 96）的情形，概率约 `1 - e^{-9.6e-5}`。

### 1.5 返回对象的形状与列

返回 `data.frame`（`:189-196`），列顺序固定为 **`x`, `y`, `ts`, `z`**：

| 列 | 含义 | 取值 |
|---|---|---|
| `x` | 电极横坐标 | 整数 1…96 |
| `y` | 电极纵坐标 | 整数 1…96 |
| `ts` | 到达时刻 | 连续（`t₀ + r/v + e`） |
| `z` | 真值标签 | `1` = 信号（`:91-92`），`0` = 噪声 |

* 行数 = 保留信号数 + 噪声点数；`:94-95` 按 `ts` **升序**重排并将 `row.names` 重置为 `1:n`。
* `visualize2d:663` 用 `as.matrix(sim.result)` 得到 **9216 × 4** 的数值矩阵 `sim.matrix`；后续只用第 1/2/3 列（x, y, ts），第 4 列（z）不参与拟合。**陷阱**：`as.matrix` 对 data.frame 会做统一类型提升，只要有一列是字符型就整张矩阵变字符型（T31）。
* `z` 只是"是否被保留"的标签，**不等于**"是否真的缺失"的计数：真实缺失数无法从返回对象恢复。

### 1.6 真值（ground truth）的编码位置

模拟器**从不返回**几何真值。真值由 `visualize2d` 用自己的硬编码常量**另行拼装**：

```r
# :659   x = 48; y = 48; u.x = 1; u.y = 1; ts = 2; ratio = 10; miux = 1; miuy = miux;
# :728   return(list('truth' = c(x, y, miux, ts), 'pp' = pp.estimate,
#                    'mean.travelling.time' = mean.travelling.time))
```

⇒ **truth 向量 = `c(x0, y0, 速度v, t0)` = `c(48, 48, 1, 2)`**（`:728`）。
第 3 个分量是**速度**（`miux = 1`），而估计量 `pp` 的第 3 槽存的是**慢度**（见 §2.1/T1），因此比较时必须先取倒数。

其余"真值"型的量：

| 量 | 位置 | 说明 |
|---|---|---|
| 缺失概率 `p` | `:658`（`theta$p`） | 真实缺失概率；但 `pp` 里**没有**对 `p` 的估计，图上只用 `z` 计数 |
| 噪声率 `λ`（`noise.freq`） | `:658`（`theta$lambda`） | 真值；同样**没有**回估 |
| `sigma` | `:658`（`theta$sigma`） | 真值；图上以 `sigma/mean.travelling.time` 作 x 轴（mode 3） |
| 信号/噪声计数 | `:667-668` | `num.true.signal = length(which(z %in% 1))`，`num.true.noise = length(which(z %in% 0))` |
| 平均传播时间 | `:672` | `mean(sim.result$ts[true.signal.indices])` ≡ `mean.travelling.time` |

**⚠️ 不确定**：`:672` 的 `mean.travelling.time` 是**被保留信号**（`z==1`）的 `ts` 均值，而不是真值 `t₀ + mean(r)/v`；`p` 大时二者会有偏差（保留是随机子集，偏差应当是二阶的，但不要假设无偏）。理论上对 96×96、中心 (48,48) 的均匀网格，`E[r] = 96·(√2 + ln(1+√2))/6 ≈ 36.73`，故 `mean.travelling.time ≈ 2 + 36.73 = 38.73`（离散网格会有极小偏移）。

### 1.7 遗留的 1D 模拟器（死代码，仅供完整性）

* `stomach.sim.one`（`:32-58`）：`dt = rnorm(1, miu, sigma)`，`while (within.range(last(x)+u.x, last(ts)+dt))` 逐步推进（`:41-46`），随后对每个位置以 `runif(1,0,1) > p` 决定是否保留（`:51-56`）。
* `stomach.sim`（`:60-97`）：`SEED_BASE = 1000000`（`:62`），种子 `= 1e6·LIMIT·time.max·ts·x·u.x·p·noise.freq·miu·sigma`；多次激发由 `rexp(1, 1/gap)` 间隔（`:73`）；噪声生成与 `stomach.sim2d` 相同（`:77-87`）；返回 `x, ts, z` **三列**（`:89-92`，无 `y`）。
* 唯一调用在 `visualize:250`（`stomach.sim(30, 150, 1, 1, 1, p, lambda, miu, sigma, 20)`），而 `visualize` 本身无人调用。此处种子 `= 4.5e9·p·λ·μ·σ`，`p·λ·μ·σ ≥ 0.48` 时同样会撞上整数溢出（T3）。

---

## 2. 模型与损失函数 / Model & loss functions

统一记号（与 R 的全局量一一对应）：

* 网格下标 `i, j ∈ {1,…,96}`；`x.grid[i,j] = i`，`y.grid[i,j] = j`（`:372-373`）；`T[i,j] = ts.grid[i,j]`，未观测格为 `0`（`:358`）。
* `r_ij(x0,y0) = norm.xy = √((i−x0)² + (j−y0)²)`（`:375-377`）。
* `1{·}` 为指示函数；R 中 `(T > 0)` 是逻辑矩阵，逐元素乘进 `sum()` 即得掩码效果。

### 2.1 四个损失函数：公式 + 参数语义

| 函数 | 行 | 公式 | `v` 的语义 |
|---|---|---|---|
| `cone.model` | `:290-301` | `L = Σ_{i,j} [ r_ij/v − (T_ij − t0) ]² · 1{T_ij>0}` | **速度** |
| `cone.model.inverted` | `:303-314` | `L = Σ_{i,j} [ r_ij·v − (T_ij − t0) ]² · 1{T_ij>0}` | **慢度 `1/速度`** |
| `cone.model.inverted.all` | `:316-325` | `L = Σ_{k=1..n} [ r_k·v − (t_k − t0) ]²`（`r_k` 由第 1、2 列算，`t_k` 第 3 列；**无掩码**） | 慢度 |
| `cone.model.square` | `:342-355` | `L = Σ_{i,j} [ r²_ij/v² − (T_ij − t0)² ]² · 1{T_ij>0}` | **速度** |

参数装载方式完全相同（每行开头一句）——`x0 = p[[1]]; y0 = p[[2]]; v = p[[3]]; t0 = p[[4]]`。这就是任务里说的"槽名与含义不符"：

> **T1（最重要）**：同一文件里 `v` 有两种互不相容的含义。`cone.model` / `cone.model.square` 里 `v` 是**速度**（做除法）；而驱动器真正使用的 `cone.model.inverted.all` 里 `v` 是**慢度**（做乘法，`t = t0 + v·r`）。`hybrid.optim` 的返回值因此是 `(x0, y0, 慢度, t0)`。`relerror.plot:793` 用 `1/pp.estimates[,4]` 把它换回速度，truth 的第 3 分量 `miux = 1` 也是速度（`:728`）。移植时建议直接命名 `slowness`，并在报告端取倒数。

数值/退化性质：
* `cone.model.inverted.all`（驱动器实际使用的那个）**不含 `1/r`**，所以在 `r = 0`（中心恰好落在某个电极上）**不会**奇异。反之 `cone.model*`（网格版）与解析梯度/海森阵含 `r^{-1}`、`r^{-3}`，`r=0` 会给出 `Inf`/`NaN`（T8）。
* `(ts.grid > 0)` 掩码：因为未观测格存的是 `0`，用 `> 0` 而不是 `!= 0`。若某次真实到达时刻恰好 ≤ 0（本文件不可能），该格会被静默丢掉。移植时应写成 `mask = ts > 0`（元素级），**不要**写成 `ts != 0`。

辅助包装函数：

| 函数 | 行 | 作用 |
|---|---|---|
| `cone.model.inverted.xy(p, ts.grid, v, t)` | `:332-335` | `pp = c(p[[1]], p[[2]], v, t)` 后调用 `cone.model.inverted`。**死代码**（`:327-330` 的对应 `vt` 版被注释）。 |
| `cone.model.inverted.all.xy(p, sim.matrix, v, t)` | `:337-340` | 同上，调用 `.all` 版。**这是 `grid.optim` 的实际 callback**：`p` 只含 `(x0, y0)` 两个自由量，`v, t0` 由外层固定。 |

### 2.2 `convert.to.grid`（死代码，`:357-368`）

```r
ts.grid = array(0, c(LIMIT, LIMIT))          # :358 ← 用的是全局 LIMIT，不是形参 limit（T21b）
for (x in 1:limit) for (y in 1:limit) {      # :359-360
   tss = sim.result$ts[which((sim.result$x == x) & (sim.result$y == y))]   # :361
   for (ts in tss) ts.grid[x, y] = ts        # :362-364 ← 后写覆盖先写
}
```

* 语义：把观测矩阵栅格化成 96×96 的"每格一个时刻"数组，空格为 0。
* **覆盖规则**：因为输入 data.frame 已按 `ts` 升序（`:195`），同一格内**最后写入的是最大的 `ts`** ⇒ 每格保留 `max(ts)`。
* R 的矩阵下标是 `[行, 列]`，`:363` 用 `[x, y]`；配合 `x.grid[i,j] = i`（列主序填充）才自洽。**用 NumPy 的行主序 (C order) 重排会静默地把波前转置**（T4）。
* 唯一调用点在 `visualize2d:670`，**已被注释**。移植可实现但默认不用（驱动器走 `cone.model.inverted.all` 而不是网格路径）。

### 2.3 解析梯度 `gra`（`:493-500`）与其 4 个分量

`gra(p, ts.grid)` 返回长度为 4 的向量 `[dx0, dy0, dv, dt0]`。**✅ 已验证**：它是 `cone.model`（速度版）的**精确**梯度——与中心差分对比，最大相对偏差 `7.0e-10`；与 `cone.model.inverted` 的梯度对比偏差 `9.0`，完全不同。

设 `r = norm.xy(x0,y0)`，`T = ts.grid`，`m = 1{T>0}`，`Σ ≡ Σ_{i,j}`：

| 分量 | 行 | 精确公式 |
|---|---|---|
| `dx0` | `:380-387` | `∂L/∂x0 = −2 Σ [ 1/v² − (T−t0)/(v·r) ] (x.grid − x0) · m` |
| `dy0` | `:389-396` | `∂L/∂y0 = −2 Σ [ 1/v² − (T−t0)/(v·r) ] (y.grid − y0) · m` |
| `dv` | `:398-405` | `∂L/∂v = −2 Σ [ r²/v³ − (r/v²)(T−t0) ] · m` |
| `dt0` | `:407-413` | `∂L/∂t0 = +2 Σ [ r/v − (T−t0) ] · m` |

R 里的写法是 `(norm.xy(x0,y0))^(-1)` 即 `1/r`，且 `^` 在 R 中对**矩阵**是逐元素幂（不是矩阵幂）——NumPy 里必须用 `**`/`np.power`，**绝不能**用 `np.linalg.matrix_power` 或 `@`（T21 的兄弟陷阱）。常数因子 `-2` / `+2` 写在 `total` 之外（`:385`,`:403`,`:411`）。

### 2.4 解析海森阵 `hes`（`:502-515`）与 10 个二阶导

`hes` 组装一个 4×4 **对称**矩阵，只算了 10 个独立元素（`:505-513` 用 `a[1,2] = a[2,1] = ...` 同时赋值）。**✅ 已验证**：与 `cone.model` 的数值海森阵对比，最大相对偏差 `2.3e-7`（差分误差量级），与 `cone.model.inverted` 对比偏差 `2.34`。因此 `gra/hes` 与 `cone.model` **是自洽的一对**，牛顿迭代（若使用）不需要线性搜索的近似修正。

| 元素 | 行 | 精确公式（`m = 1{T>0}`） |
|---|---|---|
| `dx0x0` | `:415-422` | `∂²L/∂x0² = 2 Σ [ (x−x0)²(T−t0)/(v r³) + 1/v² − (T−t0)/(v r) ]·m` |
| `dy0y0` | `:424-431` | `∂²L/∂y0² = 2 Σ [ (y−y0)²(T−t0)/(v r³) + 1/v² − (T−t0)/(v r) ]·m` |
| `dvv` | `:433-439` | `∂²L/∂v² = 2 Σ [ 3r²/v⁴ − 2r(T−t0)/v³ ]·m` |
| `dt0t0` | `:441-446` | `∂²L/∂t0² = 2 Σ m`（即 `2 × 有效格数`；代码写 `total = sum(ts.grid > 0)`） |
| `dx0y0` | `:448-454` | `2 Σ (x−x0)(y−y0)(T−t0)/(v r³)·m` |
| `dx0v` | `:456-462` | `2 Σ [ 2/v³ − (T−t0)/(v² r) ](x−x0)·m` |
| `dx0t0` | `:464-469` | `−2 Σ (x−x0)/(v r)·m` |
| `dy0v` | `:471-477` | `2 Σ [ 2/v³ − (T−t0)/(v² r) ](y−y0)·m` |
| `dy0t0` | `:479-484` | `−2 Σ (y−y0)/(v r)·m` |
| `dvt0` | `:486-491` | `−2 Σ r/v²·m` |

（`dx0x0` 与"教科书式"结果 `2Σ[1/v² − (T−t0)(y−y0)²/(v r³)]` 形式不同，但二者恒等：因为 `(x−x0)² + (y−y0)² = r²`。我一开始也差点判它写错，**注意别把它"修"成另一个形式**——要修就两者都验一遍。）

### 2.5 `norm.xy`、`normv`

* `norm.xy(x0, y0)`（`:375-378`）：返回 **96×96 矩阵** `((x.grid−x0)² + (y.grid−y0)²)^{1/2}`。它**隐式**读取全局 `x.grid`/`y.grid`（T21）。
* `normv(v)`（`:531`）：`norm(as.matrix(v), 'F')` = 向量的 Frobenius 范数 = 欧几里得范数 `√Σvᵢ²`。用于 `newton.cone.square:546`、`grid.optim:603`、`hybrid.optim:645`。

### 2.6 三个优化器

#### 2.6.1 `newton.cone`（`:517-529`，死代码）

```
iter.max = 200，无收敛判据，无线性搜索：
for i in 1:200:
    H = hes(p, ts.grid);  g = gra(p, ts.grid)
    p = p − solve(H, g)                 # :522 牛顿步
    print(‖g‖_F); print(cone.model(p, ts.grid)); print(p); print(H)   # :523-526
return p
```
* 纯牛顿迭代，`solve()` 无阻尼、无正定检查 ⇒ 海森阵非正定或奇异时**直接发散/报错**。
* 每次迭代打印 4 组内容（200 次迭代 = 800 个打印块）——移植时务必去掉或改为 logging。
* 在活跃路径上**不被调用**。

#### 2.6.2 `newton.cone.square`（`:533-554`，死代码，需 `numDeriv`）

```
iter.max = 200; epsilon = 1e-6
for i in 1:200:
    p.old = p
    H = numDeriv::hessian(cone.model.square, p, method="Richardson",
                          method.args=list(eps=1e-4, d=0.1,    r=4, v=2), ts.grid)   # :539-541
    g = numDeriv::grad   (cone.model.square, p, method="Richardson",
                          method.args=list(eps=1e-4, d=0.0001, r=4, v=2, show.details=FALSE), ts.grid)  # :542-544
    p = p − solve(H, g)                                        # :545
    if (‖p − p.old‖_F / ‖p.old‖_F < 1e-6) break                # :546-547
```
* 数值微分（Richardson 外推）；注意 **`hessian` 与 `grad` 用了不同的 `d` 步长**（0.1 vs 0.0001），这是原样照抄的，别"统一"它们。
* 目标函数是 `cone.model.square`（`v` = **速度**），与 `gra/hes` 无关。
* 在活跃路径上**不被调用**。

#### 2.6.3 `grid.optim` 三件套（`:556-606`）——**驱动器真正使用的优化器**

**(a) `grid.optim.find.best.point.1d(index, pp.current, num.splits, pp.lower, pp.upper, callback, ...)`（`:556-577`）**

```
num.spaces = num.splits − 1                       # = 3
min.value = 1e100;  min.pp = pp.current
for i in 1:(num.spaces − 1):                      # ⚠️ 只跑 i = 1, 2 → 每维 2 个采样点
    pp.current[index] = pp.lower[index] + (pp.upper[index] − pp.lower[index])/num.spaces · i
    if (index < length(pp.current)) 递归 index+1
    else value = callback(pp.current, ...)        # :569
    if (value < min.value) { min.value = value; min.pp = <当前点> }     # :571-574
return list(value = min.value, pp = min.pp)
```

* **每维只取 2 个内部点**（`i = 1, 2`），**不含端点**。`num.splits = 4` 与注释里"4 等分"的直觉不同：因为 `1:(num.spaces-1)` = `1:2`。
* 递归深度 = 参数个数 ⇒ 每次调用 callback **`2^(参数个数)` = 4** 次。
* **T15（严重）**：R 的赋值是 copy-on-modify：子层对 `pp.current` 的写入**不影响**父层，父层存下的 `min.pp` 是**值的拷贝**。若在 Python 里把同一个 list 递归传下去并在 `min.pp = cur` 处存引用，后续迭代会**就地改掉已存的最优点**（父层下一轮会覆写 `pp.current[index]`），返回的点是一个从未被评估过的混合向量。**实现时必须每层 `list(...)` 拷贝、并在保存 best 时再拷一次**。我的验证脚本在没拷贝时能跑出结果，但语义是错的。
* 比较用严格 `<`（`:571`）：`NaN` 参与比较恒为 `FALSE`，因此**一旦 callback 返回 `NaN`，搜索会静默退回初始点**（T7）。

**(b) `grid.optim.find.best.point(num.splits, pp.lower, pp.upper, callback, ...)`（`:579-584`）**：`pp.current = pp.lower` 后从 `index = 1` 进入上面的递归。

**(c) `grid.optim(pp.lower.start, pp.upper.start, callback, ...)`（`:586-606`）**

```
num.splits = 4; num.spaces = 3; iter.max = 1000; epsilon = 1e-8     # :587-590
for iter in 1:1000:
    pp.old = (pp.lower + pp.upper)/2                                # :593（其结果未被使用）
    best = find.best.point(num.splits, pp.lower, pp.upper, callback, ...)   # :594-595
    new.space = (pp.upper − pp.lower)/num.spaces                    # :596
    pp.lower = best$pp − new.space;  pp.upper = best$pp + new.space # :599-600
    if (‖new.space‖_F < 1e-8) break                                 # :603
return (pp.lower + pp.upper)/2                                      # :605
```

* 盒子每轮边长 × `2/3`（因为新盒子 = 最优点 ± 原边长/3）。
* 起点 `±500`（`:632`，见 §2.6.4）：边长 1000 → 需要 `1000·(2/3)^{n−1} · √2/3 < 1e-8` ⇒ `n = 62` 轮。**✅ 已验证：迭代次数恒为 62，与数据无关**（验证脚本计数 `744 = 3 × 4 × 62`）。
* ⇒ **每次 `grid.optim` 调用恰好评估 callback 62 × 4 = 248 次**（数据无关的硬常数）。
* `:605` 返回的是最终盒子的**中点**；因为 `pp.lower/upper` 是对称地建在 `best$pp` 两侧，中点恰好等于最后的最优点（浮点上可能有 1 ulp 误差）。`:593` 算出的 `pp.old` **是死代码**（`:602` 的 reldiff 被注释）。
* `epsilon = 1e-8` 作用在**盒子的半宽**上（`norm(new.space)`），不是作用在参数估计的精度上——`(x0,y0)` 的名义分辨率因此约 `1e-8`。

#### 2.6.4 `vt.optim` / `vt.optim.all` / `hybrid.optim`

**`vt.optim.all(p, sim.matrix)`（`:620-629`）——活跃**：

```r
x0 = p[[1]]; y0 = p[[2]]
y = sim.matrix[,3]                                            # ← 因变量 = 到达时刻（第 3 列）
x = as.vector(sqrt((sim.matrix[,1]-x0)^2 + (sim.matrix[,2]-y0)^2))   # ← 自变量 = 距离
vt = lm(y ~ x)                                                # :624
p[[4]] = vt$coefficients[[1]]     # 截距 → t0        :625  ← 先写第 4 槽
p[[3]] = vt$coefficients[[2]]     # 斜率 → v（慢度）  :626  ← 再写第 3 槽
return(p)
```
* 拟合模型 `t = t0 + v·r` ⇒ **截距 = t0，斜率 = 慢度**。与 `cone.model.inverted.all` 的参数化一致。
* `:625` 在长度 2 的向量上写 `p[[4]]`：R 会自动扩展到长度 4 并把 `p[[3]]` 置为 `NA`，随后 `:626` 覆盖它。Python 端直接构造 4 元列表即可，但**顺序别抄错**（先 4 后 3）。
* 用 `lm()`（公式接口）而非 `lm.fit()`：见 T7（秩亏时 `lm` 会把别名列剔除并返回 `NA` 系数，`lm.fit` 直接给出秩亏的 QR 结果，两者行为不同）。
* `vt.optim(p, ts.grid)`（`:608-618`）是同一逻辑的网格版：`y = as.vector(ts.grid)`、`x = as.vector(sqrt(...))`，再 `x = x[y>0]; y = y[y>0]`（`:612`）——**用"时刻 > 0"来筛掉空格子，且 x/y 用同一掩码**。两个 `as.vector` 都是**列主序**展平（T5）。**死代码**。

**`hybrid.optim(sim.matrix)`（`:631-650`）——活跃的交替最小化主循环**：

```
pp.lower = c(-500,-500); pp.upper = c(500,500)      # :632  中心搜索范围（远大于 96×96 网格！）
v = 1;  t = 1                                       # :633  (v,t0) 的初值
iter.max = 10000;  epsilon = 1e-8                   # :634-635
pp.old = c(1,1,1,1)                                 # :636
for iter in 1:10000:
    pp.estimate = grid.optim(pp.lower, pp.upper, cone.model.inverted.all.xy, sim.matrix, v, t)  # :638-639
    pp.estimate = vt.optim.all(pp.estimate, sim.matrix)                                        # :640
    v = pp.estimate[[3]];  t = pp.estimate[[4]]                                                # :643-644
    reldiff = ‖pp.estimate − pp.old‖_F / ‖pp.old‖_F                                            # :645
    if (reldiff < 1e-8) break                                                                  # :646
    pp.old = pp.estimate                                                                       # :647
return pp.estimate                                                                             # :649
```

* 交替结构：**固定 `(v, t0)` → 二维网格搜索 `(x0, y0)` → 线性最小二乘更新 `(v, t0)` → 判收敛**。这是论文 Algorithm 3 的骨架。
* 收敛判据是 4 维向量的**相对** Frobenius 变化 `< 1e-8`（`:645-646`），迭代上限 10000（`:634`）。
* `:636` 的 `pp.old = c(1,1,1,1)` 使**第一轮**的相对变化除以 `‖(1,1,1,1)‖ = 2`，而不是除以估计量的范数——第一轮的判据因此比其他轮宽松约 1–2 个数量级。原样保留（除非明确要"修正"）。
* `:646` 先 `break` 再 `:647` 更新 `pp.old`，所以退出时 `pp.old` 是**上一轮**的值——由于紧接着就 `return`，这个差异没有观测效果。
* `:638-639` 的 callback 是 `.xy` 包装：`p` 只含 2 个自由量 ⇒ 每次 `grid.optim` 的 callback 数 = `2² = 4`。

**✅ 已验证的收敛行为**（我用纯 Python 等价实现，在 24×24 代理数据上跑）：

| sigma | p | 数据集点数 | 外层迭代次数 | loss 评估次数 |
|---|---|---|---|---|
| 1e-6 | 0 | 576 | 2 | 496 |
| 1 | 0 | 576 | 3 | 744 |
| 1 | 0.4 | 325 | 5 | 1240 |
| 1 | 0.8 | 116 | 7 | 1736 |
| 0.5 | 0.8 | 116 | 7 | 1736 |

即 **外层 2–7 次**（`评估次数 = 外层 × 62 × 4`），远低于 10000 的上限；估计量收敛到真值附近（低噪声时 `x0,y0 → 真值` 到 1e-6 精度）。**⚠️ 不确定**：96×96 上的外层次数未实测，但结构上应落在同一量级（3–8 次）；`p` 越大、`sigma` 越大收敛越慢。移植时**保留 10000 的上限与 1e-8 的判据**，并把实际迭代次数记进日志以便核对。

---

## 3. 评估与绘图 / Evaluation

### 3.1 每个重复实验的驱动：`visualize2d`（`:652-730`）

这是图 6/7/8 的**单点实验函数**，也是 `truth` 与 `pp.estimates` 的来源：

```r
visualize2d = function(theta, plot.3d = FALSE) {          # :652
  limit = 96; time.max = limit; gap = 1e6                  # :653-657
  lambda = theta$lambda; p = theta$p; sigma = theta$sigma  # :658
  x = 48; y = 48; u.x = 1; u.y = 1; ts = 2; ratio = 10; miux = 1; miuy = miux   # :659
  sim.result = stomach.sim2d(limit, time.max, ts, x, y, u.x, u.y, p, lambda,
                             miux, miuy, sigma, gap, ratio)                     # :661-662
  sim.matrix = as.matrix(sim.result)                        # :663
  z.true = sim.result$z                                     # :664
  true.signal.indices = which(z.true %in% 1)                # :665
  true.noise.indices  = which(z.true %in% 0)                # :666
  num.true.signal = length(true.signal.indices)             # :667
  num.true.noise  = length(true.noise.indices)              # :668
  pp.estimate = hybrid.optim(sim.matrix)                    # :671
  mean.travelling.time = mean(sim.result$ts[true.signal.indices])   # :672
  pp.estimate = append(pp.estimate, c(num.true.signal, num.true.noise,
                                      mean.travelling.time))        # :673-674
  if (plot.3d) { ...rgl... }                                # :676-727
  return(list('truth'=c(x,y,miux,ts), 'pp'=pp.estimate,
              'mean.travelling.time'=mean.travelling.time))          # :728-729
}
```

**每个重复实验只产生 `pp.estimate` 一个 7 维向量**（`:673-674` 的 `append` 追加在尾部）：

| 槽 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| 含义 | `x0` | `y0` | `v`（**慢度**） | `t0` | `num.true.signal` | `num.true.noise` | `mean.travelling.time` |

`optimality.table` 随后在**最前面**插入扫描变量（`:893`/`:912`/`:933` 的 `cbind`），得到 8 列矩阵：

| 列 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 含义 | **扫描值**（λ / p / σ） | `x0` | `y0` | `v`（慢度） | `t0` | `num.signal` | `num.noise` | `mean.tt` |

### 3.2 相对误差的定义（`relerror.plot:790-802`）

```r
if (plot.value) {                       # :790  ← 本文件所有调用都走这一支
    relerror.x = pp.estimates[,2]
    relerror.y = pp.estimates[,3]
    relerror.v = 1/pp.estimates[,4]     # :793  慢度 → 速度
    relerror.t = pp.estimates[,5]
} else {                                # :795-802  ← 本文件永不进入
    relerror.x = |pp.estimates[i,2] − truth[[1]]| / truth[[1]]
    relerror.y = |pp.estimates[i,3] − truth[[2]]| / truth[[2]]
    relerror.v = |1/pp.estimates[i,4] − truth[[3]]| / truth[[3]]
    relerror.t = |pp.estimates[i,5] − truth[[4]]| / truth[[4]]
}
```

* 归一化用的 `truth = c(48, 48, 1, 2)`（`:728`）：
  * `x`：除以**真 x₀ = 48**
  * `y`：除以**真 y₀ = 48**
  * `v`：先取 `1/pp[,4]`（慢度→速度）再除以**真速度 `miux = 1`**
  * `t`：除以**真 t₀ = 2**
  即四个量都是"相对真值的绝对误差"，分母各不相同，且 `v` 那一路必须先取倒数。
* 列号是**含扫描变量之后**的列号（§3.1 的第二张表）。若误把未 `cbind` 的 7 列 `pp` 传进来，`[,4]` 会读到 `t0` 而不是 `v`——**静默算错**（T3）。
* **T26（关键语义陷阱）**：函数名叫 `relerror.plot`，但 `optimality.table` 的三处调用都把第 5 个位置参数 `plot.value` 传成 **`TRUE`**（`:897`、`:916`、`:938`），于是 `:796-801` 的相对误差分支**根本不会执行**：图上画的是**绝对估计值**（`v` 那一列是 `1/pp[,4]` = 速度），只有 `plot.value = FALSE` 时才是相对误差。**移植时若"顺手修正"成相对误差，图 7/8 的 y 轴会与论文完全不同。** ⚠️ 不确定：论文正文把图 7/8 的 y 轴描述成什么（我无法在本机读 `.work/paper.pdf`；`whd-20260915release/docs/code-paper-mapping.md:105-106` 只写了 "accuracy vs SNR / vs missing probability"）。

### 3.3 SNR 的定义（`:787-789`）

```r
for (i in 1:nrow(pp.estimates))  snr = append(snr, pp.estimates[i,6]/pp.estimates[i,7])
```

⇒ **SNR ≔ `num.true.signal / num.true.noise`**（第 6 列 ÷ 第 7 列，即 `cbind` 之后的信号观测**计数** ÷ 噪声观测**计数**）。

* 这是**计数比**，不是方差比/功率比；它只通过"噪声点数"间接依赖 `λ`，且**依赖于 `arrival.time < time.max` 这个过滤**。
* 本文件配置下 `num.signal ≡ 9216`（§1.3：全部格子都及时到达，`p=0`），`num.noise ≈ K − 1 ≈ 96·λ`。
  ⇒ mode 1 的 `lambdas = 2^((-6:16)/2) ∈ [0.125, 256]` 对应 **SNR ≈ 9216/(96λ) ∈ [≈0.375, ≈768]**（期望值，实际值随 RNG 波动；下界 `0.375` 与 `code-paper-association.md:192` 记录的论文值 `0.37` 吻合）。
* ⚠️ **未解冲突**：`code-paper-association.md:192` 记论文图 7 的 SNR 范围是 `0.37 → 3072`，而本代码的 `λ` 上界（0.125）只能给出 `≈768`。要得到 3072 需 `num.noise ≈ 3`，即 `λ ≈ 0.03`——不在 `2^((-6:16)/2)` 里。**可能**论文的 SNR 用了另一套计数方式或更早版本的 λ 网格，我无法确证，**移植时以代码为准并把这个差异写进复现说明**。
* `log10snr` 这个名字**具有误导性**：`:810` 起 `log10snr = snr`（未取对数），只有 mode 1 通过 `plot(..., log="x")`（`:850-852`）把 x 轴变成对数轴。mode 2/3 的 x 轴是**线性**的。

### 3.4 `relerror.plot` 的参数与 `plot.mode` 开关的完整对应

函数签名（`:782-784`）：

```r
relerror.plot(pp.estimates, truth, label, variables, plot.value, should.log,
              should.use.snr, params, add.text, should.use.travel = FALSE)
```

| 形参 | 作用 | 相关行 |
|---|---|---|
| `label` | `c(x轴标题, y轴标题)` | `:851-855` |
| `variables` | 取 `variables[[1]]` 作为 `i`，决定画 `ys[,i]`（1=x0, 2=y0, 3=速度, 4=t0）**以及** `truth[[i]]` | `:825`、`:844`、`:867`、`:872` |
| `plot.value` | `TRUE` = 画绝对估计值；`FALSE` = 画 §3.2 的相对误差 | `:790-802`、`:871-874` |
| `should.log` | 只影响 `plot(..., log="x")` | `:850-856` |
| `should.use.snr` | `FALSE` ⇒ x 轴改用第 1 列（扫描值） | `:816-819` |
| `params` | 点旁标注的文字（`round(params, 2)`） | `:857`、`:869` |
| `add.text` | 是否画标注 | `:868-870` |
| `should.use.travel` | `TRUE` ⇒ x 轴改为 `第1列 / 第8列`（**覆盖** `should.use.snr` 的效果，因为 `:820-822` 在 `:816-819` 之后） | `:820-822` |

`optimality.table`（`:878-1008`）里的实际调用：

| `plot.mode` | 行 | 扫描变量与网格 | 其余固定参数 | 重复数 | `visualize2d` 调用 | `relerror.plot` 位置参数（第 5 起） | 输出 |
|---|---|---|---|---|---|---|---|
| **1** | `:884-902` | `lambdas = 2^((-6:16)/2)` = `2^{-3}…2^{8}`，**23 个点**（`:885`） | `p = 0`, `sigma = 1e-6`（`:888`） | 23 | `visualize2d(theta.true)`（`:889`，`plot.3d=FALSE`） | `variable, TRUE, TRUE, TRUE, lambdas, TRUE`（`:897`）⇒ x = **SNR**（对数轴），y = 绝对估计，标 λ | `plots/plot_mode1_1..4.pdf`（`:898-900`） |
| **2** | `:904-921` | `ps = (0:8)/10` = `0.0…0.8`，**9 个点**（`:905`） | `lambda = 1`, `sigma = 1e-6`（`:907`） | 9 | 同上（`:908`） | `variable, TRUE, FALSE, FALSE, pp.estimates[,1], FALSE`（`:916`）⇒ x = **p**（线性），y = 绝对估计，无标注 | `plots/plot_mode2_1..4.pdf`（`:917-919`） |
| **3**（**提交态**，`:880`） | `:923-943` | `sigmas = (1:10)/10` = `0.1…1.0`，**10 个点**（`:924`） | `lambda = 1e-6`, `p = 0`（`:928`） | 10 | 同上（`:929`） | `variable, TRUE, FALSE, FALSE, pp.estimates[,1], TRUE, TRUE`（`:938`，**第 10 个参数**）⇒ x = **σ / mean.tt**（第1列/第8列，线性），y = 绝对估计，标 σ | `plots/plot_mode3_1..4.pdf`（`:939-941`） |
| 4 | `:945-953` | `sigmas = 10`（单点） | `lambda = 100`, `p = 0`, `sigma = 1e-6` | 1 | `visualize2d(theta.true, TRUE)`（`:950`） | —（不画） | `plots/cone_simulate.png`（`:687`） |
| 5 | `:955-959` | 无扫描（单点） | `lambda = 1`, `p = 0.1`, `sigma = 1` | 1 | `visualize2d(theta.true, TRUE)`（`:958`） | — | 同上 PNG |
| 6 | `:962-974` | `ps = (0:8)/10` | `lambda = 1e-7`, `sigma = 1e-6` | 9 | `visualize2d(theta.true)`（`:966`） | `4, TRUE, FALSE, FALSE`（`:973`，**只有 7 个参数**） | **会报错**（见下） |
| 7 | `:976-988` | `ps = (0:8)/10` | `lambda = 1`, `sigma = 1e-6` | 9 | `visualize2d(theta.true)`（`:980`） | `4, TRUE, FALSE, FALSE`（`:987`，**7 个参数**） | **会报错** |
| 8 | `:990-996` | `lambdas = 1`（单点） | `lambda = 1`, `p = 0`, `sigma = 1e-6` | 1 | `visualize2d(theta.true, TRUE)`（`:994`） | — | 同上 PNG |

* **T24：mode 6/7 会崩。** 它们只传 7 个位置参数（`:973`、`:987`），`params` 与 `add.text` 缺失；`:857` 的 `params = round(params, digits = 2)` 在绘图**之前**无条件求值 ⇒ `argument "params" is missing, with no default`。这两个 mode 从未产出过图，**不要**把它们当成"参考实现"。
* **T25：`relerror.plot:828-838` 的 `variables == 4` 分支是死代码**：`:845-849` 无条件重算 `ymin/ymax`（`min/max(ys[,i], truth[[i]])` 各外扩 10% 的极差）。该分支只剩两个 `print` 副作用（`:830`、`:832`）。
* **T28：mode 1 的 x 轴**是**线性 SNR 值**（`snr`，`~0.375…768`）画在对数轴上（`log="x"`），不是 `log10(snr)`。
* **T27：mode 3 的 x 轴**是 `σ/mean.travelling.time`（`:820-822` 覆盖 `:816-819`），`mean.tt ≈ 38.7`（§1.6）⇒ x ∈ 约 `[0.0026, 0.026]`；标签在 `:936-937`。同时 mode 3 的 `should.log = FALSE` ⇒ **全线性轴**。
* 每个 mode 对 `variable = 1..4` 各出一张 PDF，文件名 `plots/plot_mode<mode>_<variable>.pdf`（`:898-900`、`:917-919`、`:939-941`）。`truth.estimate.pair` 是**循环结束后**最后一个扫描点的真值（`:895`/`:914`/`:935` 在循环外），本文件里各点真值恒定，所以无害；若将来让每点真值不同，这是个 bug（T43）。

### 3.5 图 7 / 图 8 的轴与"汇总"方式

| 图 | 由哪个 mode 产出 | x 轴 | y 轴 | 汇总 |
|---|---|---|---|---|
| **图 6**（96×96 圆波前 3D 示意） | mode 4 / 5 / 8 的 `plot.3d = TRUE` 分支（`:676-727`） | 3D 散点：x = 电极 x，y = 电极 y，z = `ts`；噪声黑点（`plot3d:680-684`）、信号绿点（`points3d:685-686`） | — | 无汇总；`rgl.snapshot("plots/cone_simulate.png")`（`:687`）覆盖式输出 |
| **图 7**（准确度 vs SNR） | `plot.mode == 1`（`:884-902`） | SNR = `第6列/第7列`（信号计数/噪声计数），对数轴 | `ys[,variable]`：`variable=1→x0`、`2→y0`、`3→1/pp[,4]`（**速度**）、`4→t0`；另画一条绿色水平真值线（`:871-874`） | **没有任何汇总**：每个重复实验 = 1 个点，无均值/中位数/误差条/拟合线。点旁文字是所有扫描点的 λ 值（`params = lambdas`，`:897`） |
| **图 8**（准确度 vs 缺失概率） | `plot.mode == 2`（`:904-921`）（⚠️ 见下） | `p`（线性，0…0.8） | 同上 | 同上；无点旁文字（`add.text = FALSE`，`:916`） |

* **⚠️ 不确定**：`whd-20260915release/docs/reproducibility-notes.md:17-18` 与 `code-paper-mapping.md:105-106` 都把图 8 对应到 `plot.mode == 2`，而**文件里提交的值是 `plot.mode = 3`**（`:880`）。mode 3 扫的是 `σ`（测量误差 / 平均传播时间），不是缺失概率 `p`。两者产出的曲线不同，务必以论文图注为准再决定移植哪一支。
* 唯一出现的"均值"是 `mean.travelling.time`（`:672`），它只当 mode 3 的 x 轴分母，**不是**对重复实验的汇总。
* 图 7/8 的纵轴语义再次强调：因为 `plot.value = TRUE`，画的是**绝对估计量**；错误条/离散度一概没有。

### 3.6 `hough` 与 `point.angle.to.line`（死代码，`:732-780`）

只在 `visualize`（`:244-288`，本身无人调用）里被用到；与图 6/7/8 无关。若确需移植，行为如下：

**`point.angle.to.line(x, y, angle)`（`:732-743`）**
```
distance.shift = 2
radian = angle/180·π;  slope = tan(radian)
distance = |y − x·slope| / √(slope² + 1)      # :736  点到"过原点的直线"的垂距
distance.shifted = distance · 2               # :737
distance.rounded = as.integer(distance.shifted)   # :738 ← 向零截断（不是四舍五入，等价于宽度 0.5 的分箱）
key = paste(angle, distance.rounded, sep=",")     # :739 ← 字符串键
debug.string = sprintf("ts=%0.2f,x=%d,d=%0.2f,ds=%0.2f,k=(%s)", ...)   # :740-741
```
* `:740` 的格式串把两个参数命名为 `ts=` 与 `x=`，但形参是 `(x, y)`：**调用时第一个实参装的是"时间"**（`:278-280` 传 `ts[...]`、`:256` 传 `hough(ts, x)`）。→ 参数顺序陷阱 T22。
* 直线族被强制过**原点**（因此只有"角度+截距"中的角度是有效的几何参数，`distance` 实际被当成一个与点的距离耦合的量）——这是早期 1D 代码的产物，**不是**论文 §2.2 的随机 Hough 平面检测。

**`hough(xs, ys)`（`:745-780`）**
```
for i in 1:num.data:  for angle in (1:36)*5:   # :752 ← 5° 步长（注释却写 "10 degree intervals"）
    把 (i, debug.string) 追加到 hash 桶 angle.distance.to.points[[key]]   # :754-759
num.points.threshold = 15                       # :763
遍历所有 key（hash 迭代序）：若 nrow > 15 → rbind 进 indices.above.threshold 并 best.angle = angle   # :766-777
return list(signal.guess = data.frame(indices.above.threshold), best.angle = best.angle)             # :778-779
```
* `(1:36)*5` = 5,10,…,180，共 36 个角度；**注释与代码不符**（T20）。
* `best.angle` 是**最后一个**超阈值的 key（`:775`），取决于 `hash` 包的键迭代顺序 ⇒ **不是确定值**（T21h）。且只保留一个角度，而不是一组平面。
* `signal.guess` 的列结构：`rbind()` 把一个 list 并入 data.frame，只用到 `nrow()`（`:768`）与 `unlist(...$indices)`（`visualize:259`）。⚠️ 我没有实测 `rbind(data.frame, list(...))` 的确切列名；移植时只需保证 `indices` 与 `label` 两列存在。
* `print(num.points)`（`:770`）会为每个超阈值 key 打印一行。

---

## 4. 计算成本 / Cost profile

### 4.1 规模数字

| 量 | 值 | 依据 |
|---|---|---|
| **提交态（`plot.mode = 3`）的重复实验数** | **10** | `sigmas = (1:10)/10`（`:924`），每点 1 次 `visualize2d`（`:929`） |
| 论文图 7（mode 1）的重复数 | 23 | `2^((-6:16)/2)`（`:885`） |
| 论文图 8（mode 2）的重复数 | 9 | `(0:8)/10`（`:905`） |
| 全部 8 个 mode 的总重复数 | **63**（23+9+10+1+1+9+9+1） | `:885` `:905` `:924` `:947` `:957` `:963` `:977` `:991` |
| **每个数据集的点数** | **9216**（= 96×96，mode 3 无噪声点） | `limit = 96`（`:653`）、双重循环 `:109-110`、全部 `arrival.time < 96`（`:131`）、`p = 0` ⇒ 全保留（`:132`）、`lambda = 1e-6` ⇒ 噪声点数 `0`（`:177-184`） |
| 其它 mode 的点数 | 信号 ≈ `9216·(1−p)`，噪声 ≈ `96·λ`（期望） | 同上的计数规则 |
| 每格 RNG 调用 | 1 次 `rnorm`（`:123`）+ 1 次条件 `runif`（`:132`） | §1.3 |

### 4.2 每次拟合的损失函数评估次数

* `hybrid.optim` 的外层：实测 **2–7 次**（§2.6.4，代理数据），上限 10000（`:634`）。
* 每次外层 = 1 次 `grid.optim` + 1 次 `lm`（9216 点、2 参数，代价可忽略）。
* 每次 `grid.optim`：**恒为 62 轮 × 4 次 callback = 248 次**（✅ 数值验证；62 由起点半宽 1000 与 `ε = 1e-8` 的几何收缩唯一确定，与数据无关）。
* ⇒ **每次拟合 ≈ 3 × 248 ≈ 744 次损失评估**（区间 496–1736），每次评估遍历 **9216 个点**：
  `744 × 9216 ≈ 6.9×10⁶ 点次`，每次点次包含 2 次平方、1 次开方、2 次减、1 次乘、1 次平方、1 次加 ≈ 8–10 flops ⇒ **单次拟合约 6×10⁷ flops**。
* 上限情形（打到 10000 轮）：`10000 × 248 = 2.48×10⁶` 次评估 ≈ `2.3×10¹⁰ 点次`，但这**不会**发生（实测 2–7 轮就收敛）。

### 4.3 主导的内层调用

**`cone.model.inverted.all`（`:316-325`）被 `cone.model.inverted.all.xy`（`:337-340`）包装，后者由 `grid.optim.find.best.point.1d` 的内层循环（`:569`）在每个候选点上调用**。整条链路的 99.9% 时间花在这里；`vt.optim.all`（`:620-629`）的 `lm` 只有 2–7 次/拟合，可以忽略。移植时**只要把这一处的 9216 点求和向量化**，其余部分怎么实现都无所谓。

数据集生成（`:100-198`）本身也要遍历 9216 个格子并做 9216 次 `rnorm`，但相对拟合可忽略（每次实验只有一次）。

### 4.4 运行时间量级

| 实现方式 | 提交态（10 个重复） | 全部 8 个 mode（63 个重复） |
|---|---|---|
| **NumPy 向量化**（推荐） | 7440 次评估 × 9216 点 ≈ 6.9×10⁷ 点次；单次评估（9216 元素的 sqrt/算术，约 5 个临时数组）约 30–100 µs ⇒ **纯计算 0.2–0.7 s**，含 Python 开销与出图 **< 1 分钟** | ≈ 4.7×10⁴ 次评估 ⇒ **纯计算 2–5 s，含绘图 1–3 分钟** |
| **纯 Python 双重循环**（把 R 逐行直译） | 6.9×10⁷ 点次 × 每点次约 10 次解释器级运算 ≈ **数十秒到数分钟** | 63/10 倍 ⇒ **10–60 分钟** |
| R 原版（参考） | `sum()` 在 C 层循环，每次拟合约 0.5–3 s ⇒ 10 个重复约 **10–30 s** | 约 **1–5 分钟**（另有大量 `print` 输出开销） |

**结论：量级是"分钟"，不是"小时"，更不是"天"。**
* 提交态 mode 3：向量化实现**秒级**，直译实现**分钟级**。
* 最有价值的优化：向量化 `cone.model.inverted.all`；其次是缓存"同一 `(x0,y0)` 组合"的评估结果（62 轮 × 4 点中实际只有 4 个新组合/轮，无重复，故缓存收益为零——**不要**为缓存花功夫）。
* 若把 `grid.optim` 的 62 轮换成连续的 `scipy.optimize` 或把交替最小化换成 `scipy.optimize.least_squares`，可再快 1–2 个数量级，但那**会改变数值结果**（网格分辨率、收敛判据都不同），属于"新实验"而非"移植"。

---

## 5. 图形与 headless 依赖 / Graphics

### 5.1 是否需要交互式/OpenGL 设备

| 位置 | 调用 | headless 影响 |
|---|---|---|
| `:3` | `require(rgl)` | **加载期**依赖。无 OpenGL/显示设备时可能加载失败；macOS 上通常需要 XQuartz。⚠️ 取决于 rgl 版本，`rgl.useNULL()` 可让部分功能在无窗口模式下工作，但 `plot3d` 仍需设备。 |
| `:680`、`:685`、`:687`（`visualize2d` 的 `plot.3d` 分支，`:676-727`） | `plot3d()` → `points3d()` → `rgl.snapshot("plots/cone_simulate.png")` | **硬失败**。只在 `plot.3d = TRUE` 时触发：mode 4（`:950`）、mode 5（`:958`）、mode 8（`:994`）。这是**图 6** 的唯一产出路径。 |
| `:898-900`、`:917-919`、`:939-941` | `dev.copy(pdf, ...)` + `dev.off()` | 需要存在"当前设备"。Rscript 下默认设备是 `pdf()`，一般能工作；但在某些 headless 配置下 `dev.copy` 会产出空白页。**建议移植时直接 `savefig`。** |
| `:284` | `identify(...)`（在 `visualize` 内，死代码） | 交互式点选，headless 直接失败。 |
| `:1` | `options(error=recover)` | 非交互会话里出错会尝试从 stdin 读选择，可能**挂起**。 |

**推论**：在 headless 机器上
* `plot.mode ∈ {1, 2, 3, 6, 7}` **可以**跑完（只要 `rgl` 能加载；若 `require(rgl)` 失败会打警告但不中断）；
* `plot.mode ∈ {4, 5, 8}`（图 6 的 3D 图）**必须**有 OpenGL 设备；
* 移植版建议：3D 用 `matplotlib` 的 `mplot3d`/`plotly`/`pyvista` 直接出 PNG，完全绕开 rgl。

### 5.2 其它环境要求

* 需要预先存在 `plots/` 目录（`:11-13` 会创建）。
* `hash`（`:2`）只服务 `hough`；`numDeriv`（`:4`）只服务 `newton.cone.square`。**活跃路径上二者都不需要** ⇒ Python 移植的最小依赖是 `numpy`（+ `matplotlib` 出图），与 `pyproject.toml` 现有的 `numpy/scipy/pandas` 一致。

---

## 6. 陷阱清单 / Traps

按"复现风险"排序。括号内是相关行号。

### A. 索引与数据布局

* **T4. 列主序 vs 行主序。** `x.grid = array(1:LIMIT, c(LIMIT,LIMIT))`（`:372`）是**列主序**填充 ⇒ `x.grid[i,j] = i`；`y.grid = t(x.grid)`（`:373`）⇒ `y.grid[i,j] = j`；`ts.grid[x,y]`（`:363`，R 的 `[行,列]`）与之一致。用 `numpy.arange(1,97).reshape(96,96)`（C 序）+ `[:,None]`/`[None,:]` 广播时**必须显式核对"谁是行"**，否则波前被转置、`x0/y0` 互换，而误差看起来只是"精度差一点"。
* **T5. `as.vector` 是列主序展平。** `vt.optim:610-611` 的两个 `as.vector` 必须用**同一种**展平顺序（NumPy 里是 `order='F'`，或干脆直接 `ravel` 两个数组时保持同一约定），否则距离与时刻错配。
* **T6. 1-based 全部坐标。** 电极坐标 1…96、网格下标 1…96、`array(0, c(LIMIT,LIMIT))` 的空格哨兵是 `0`。真值中心是 `(48,48)`，而网格的几何中心是 `48.5` ——**这是"x0=y0=48"的由来，不要"顺手"改成 47.5 或 48.5**。
* **T30. `(ts.grid > 0)` 掩码严格大于。** 用 `> 0`（`:297`、`:310`、`:349`、`:384`…）而不是 `!= 0`。移植时 `mask = ts > 0`。
* **T31. `as.matrix(data.frame)` 的类型提升。** `visualize2d:663` 把 4 列 data.frame 变成数值矩阵；若某列被解析成字符，整张矩阵变字符。Python 端应显式构造 `float64` 的 `(n,4)` 数组。

### B. 参数语义（"槽名与含义不符"）

* **T1. `v` 的双重含义。** `cone.model`/`cone.model.square` 里是**速度**（除法），`cone.model.inverted*` 里是**慢度**（乘法）。驱动器返回的是慢度；`relerror.plot:793` 用 `1/pp[,4]` 换回速度；truth 第 3 分量是速度（`:728`）。
* **T2. truth 的顺序是 `(x0, y0, 速度, t0)`**（`:728`），而 `pp` 的顺序是 `(x0, y0, 慢度, t0, n_sig, n_noise, mean.tt)`（`:673`）。
* **T3. 列号在 `cbind` 之后整体右移一位。** `relerror.plot` 用 `[,2..5]` 取参数、`[,6]/[,7]` 算 SNR、`[,1]` 与 `[,8]` 作 x 轴（`:791-800`、`:788`、`:817`、`:821`），这是**插入了扫描列之后**的编号（`:893`/`:912`/`:933`）。直接把 `visualize2d` 的 7 维 `pp` 传进去会静默读错列。
* **T41. 信号/噪声计数的槽序不能颠倒：** 第 5 槽是信号数、第 6 槽是噪声数（`:673`）；SNR 是**信号 ÷ 噪声**（`:788`）。反过来会把图 7 的 x 轴做倒数。
* **T36. 单位：** 空间是"网格单位"，`v = 1` 表示"每单位时间 1 格"，`t0 = 2`。论文里"v=1"对应慢度 1（速度 1）。`ratio = 10`（`:659`）在本模拟器里**没有作用**。
* **T34. 形参 `u.x, u.y, ratio, noise.freq` 在 `stomach.sim.one2d` 内部完全未被使用**（`:100-139`；`ratio` 只出现在被注释的 `:148`）。别去"用上"它们。
* **T35. `theta$lambda` 是噪声率，不是波的发生率。** 它被绑到形参 `noise.freq`（`:661`，第 9 个位置）；波的发生率由硬编码的 `gap = 1e6` 控制（`:657`）。

### C. 随机性

* **T10. `p = 0` ⇒ 种子恒为 0 ⇒ 重复实验不独立。** 种子表达式含因子 `p`（`:160-161`），mode 1/3/4/8 全用 `set.seed(0)`。mode 3（提交态）的 10 个"重复"共享同一条 RNG 流，误差序列只差一个 `sigma` 缩放，噪声点数恒为 0。**这是移植时最需要向项目主人确认的决策点**：严格复现 ⇒ 也共享种子；追求统计意义 ⇒ 显式给每个扫描点独立种子，并在文档里声明偏离（这会改变图的离散度）。
* **T11. 种子整数化与溢出。** `set.seed()` 对浮点做向零截断（⚠️ 未实测）；`as.integer` 超过 `2^31−1` 会给 `NA` 并使 `set.seed` 报错（阈值 `p·noise.freq·sigma < 0.5057`）。建议显式 `int(...) % 2**31`。
* **T12. `numpy` 无法复现 R 的 RNG。** R 默认 Mersenne-Twister + Inversion 正态 + （≥3.6）Rejection 抽样；`sample()` 的算法在 R 3.6 改过（`sample.kind`）。**移植只能是统计等价，不可能逐位相同**——这一条要写进移植包的 README，避免以后有人拿"数字不一致"当 bug 追。
* **T12b. RNG 调用顺序必须原样保留**（§1.3 的 9 步）：每格 1 次 `rnorm`（无条件）→ 条件 `runif` → 波间隔 `rexp` → 噪声 `rexp` 流 → 两次 `sample`。
* **T33. `arrival.time < time.max` 是严格小于，且在本配置下恒为真**（`:131`，最大到达时刻 ≈ 68.5 < 96）。不要把它"修"成取模/回绕。
* **T35b. `sample(1:96, 0, replace = TRUE)`**（`:186-187`，mode 3 的常态）：返回 `integer(0)`。⚠️ 零长度抽样是否消耗 RNG 状态随 R 版本而异；但这两次调用是本实验**最后**的 RNG 使用，故不影响观测数据。
* **T29. `noise.ts` 首尾裁剪**（`:180-184`）：`noise.ts` 以常量 `0` 开头且以一个 `> time.max` 的增量结尾，两者都要去掉；若 `length == 2` 则结果为空向量（0 个噪声点）。

### D. 优化器与退化情形

* **T15. `grid.optim.find.best.point.1d` 的拷贝语义。** R 的 copy-on-modify 保证：(i) 子层递归写 `pp.current` 不影响父层；(ii) `min.pp` 存的是**值快照**（`:573`）。Python 里共享同一个 list 会导致"保存的最优点被后续迭代就地改写"，返回一个**从未被评估过的混合向量**。必须逐层 `list(...)` 拷贝。
* **T16. 每维只采 2 个内部点**（`:561` 的 `1:(num.spaces-1)` = `1:2`），端点不参与；`num.splits = 4` 的"4"并不对应 4 个采样点。
* **T17. `grid.optim` 返回的是盒子中点**（`:605`），只在 `lower/upper` 对称构造于 `best` 两侧时才等于最优点；`:593` 的 `pp.old` 是死代码。
* **T18. `hybrid.optim` 第一轮的判据除以 `‖(1,1,1,1)‖ = 2`**（`:636`、`:645`）；`break` 在更新 `pp.old` 之前（`:646-647`）。
* **T19. 中心搜索范围是 `[-500, 500]`**（`:632`），**远超** 96×96 网格；`(x0,y0)` 可能落到网格外甚至负值，`cone.model.inverted.all`（无 `1/r`）对此完全容忍。
* **T7. `lm` 的秩亏与 `NaN` 传播链。** `vt.optim.all:624` 用 `lm`（而非 `lm.fit`）：若所有点对当前中心的距离**完全相同**（设计矩阵秩亏），`lm` 会剔除别名列并返回 `slope = NA`；随后 `NaN` 进入 `grid.optim`，而 `:571` 的 `x < min.value` 对 `NaN` 恒为 `FALSE` ⇒ 搜索静默退回初始点（`pp.lower`）。本文件的数据（9216 个不同距离）几乎不可能触发，但**移植时要在 `vt` 里加"距离方差为 0"的显式分支并记录**，否则出问题时极难定位。
* **T8. `r = 0` 的奇异性**只影响 `cone.model*`/`norm.xy`/`gra`/`hes` 这条（死代码）路径：`(norm.xy)^(-1)`、`^(-3)` 在中心恰好落在电极上时给出 `Inf` → `NaN`。活跃路径（`cone.model.inverted.all`）无此问题。
* **T32. `distance == 0` 的浮点精确比较**（`:125`）：模拟器用它在中心格上走"无传播时间"分支。因为 `x = y = 48` 是整数且 `i, j` 是整数，这个比较是安全的；但移植时不要改成 `abs(distance) < eps`（会改变哪些格走哪一支）。
* **T34b. R 的 `^` 对矩阵是逐元素幂**（`norm.xy(x0,y0))^(-1)`）：NumPy 里对应 `** -1`，**不是** `np.linalg.inv`/`np.linalg.matrix_power`/`@`。

### E. 评估与绘图

* **T26. `plot.value = TRUE` 让 `relerror.plot` 画的是绝对估计值**，相对误差分支（`:796-801`）在三个活跃 mode 里都不可达（`:897`、`:916`、`:938`）。
* **T24. mode 6/7 因缺 `params`/`add.text` 参数而报错**（`:973`、`:987` → `:857`）。
* **T25. `variables == 4` 分支是死代码**，`ymin/ymax` 在 `:845-849` 被无条件重算。
* **T27. mode 3 的 x 轴是 `σ / mean.travelling.time`**（`:820-822`），不是 σ 本身，也不是 SNR；全线性轴。
* **T28. mode 1 的 x 轴是线性 SNR 画在对数轴上**（`:810`、`:852`），变量名 `log10snr` 有误导性。
* **T43. `truth` 用的是循环结束后最后一个扫描点的真值**（`:895`、`:914`、`:935`）。本文件各点真值恒定故无害；若让每点真值不同，这是 bug。
* **T29b. `:866` 的 `print(log10snr)`、`:830/:832` 的 `print(ymin/ymax)`、`:891/:910/:931/:968/:982` 的 `print(<扫描值>)`** 都是副作用输出，移植时改成 logging，否则 stdout 会被几万行数字淹没（`newton.cone` 的 `:523-526` 更夸张）。
* **T9. 归一化分母各不相同**：`x→48`、`y→48`、`v→1`（且先取倒数）、`t→2`（`:797-800`）。没有统一的"相对误差"定义。
* **T37. `mean.travelling.time` 只在 `z == 1` 的行上取均值**（`:672`），且它是**被保留**子集的均值，不是真值 `t0 + E[r]/v`。

### F. 代码工程

* **T21. `x.grid`/`y.grid`/`LIMIT` 是全局量**，被 `cone.model`、`cone.model.square`、`cone.model.inverted`、`norm.xy`、`dx0`…`hes` 隐式捕获（`:371-373`）。这些函数**不是纯函数**：Python 端必须显式传网格。
* **T21b. `convert.to.grid(sim.result, limit)` 用全局 `LIMIT` 分配数组**（`:358`），忽略自己的 `limit` 形参；`stomach.sim.one2d` 内部的 `within.range`（`:103-105`）同样引用全局 `LIMIT`（而 `stomach.sim.one` 的 `:34-36` 正确闭包了形参 `LIMIT`）。两处都是潜在 bug，都在死代码里。
* **T13. `require()` 不中断。** 缺 `rgl`/`hash`/`numDeriv` 时脚本继续执行到真正的调用处才失败。
* **T23. `identify()` 与 `cbind` 索引**（`:284-287`）：交互式 + 长度不等的 `cbind` 循环复用，属死代码。
* **T20. `for (angle in (1:36)*5)`（`:752`）是 5° 步长**，而注释 `:751` 写的是 "10 degree intervals"。
* **T22. `point.angle.to.line` 的实参顺序**（`:278-280`、`:256`）：形参名是 `(x, y)`，实际传入的是 `(时间, 坐标)`——移植时不要按参数名推断含义。
* **T21h. `hough$best.angle` 不确定**（`:766-776`）：取的是 `hash` 键迭代序里最后一个超阈值 key，依赖哈希实现；且只保留一个角度而不是一组平面。
* **T38. `point.angle.to.line:738` 用 `as.integer` 截断**（分箱宽度 0.5），不是四舍五入。
* **T39. `optimality.table()` 在 `:1010` 被顶层调用**，`source()` 即跑全流程。移植时改成 `if __name__ == "__main__":`。

---

## 7. 尚不确定、需与项目主人确认的点

| # | 问题 | 影响 |
|---|---|---|
| U1 | **图 8 到底是 mode 2（`p` 扫描）还是 mode 3（`σ` 扫描）？** 代码提交值是 3（`:880`），发布文档说是 2（`reproducibility-notes.md:17-18`）。 | 决定移植哪一支扫描；两条曲线的 x 轴语义完全不同。 |
| U2 | 图 7/8 的纵轴在论文里是**绝对估计值**还是**相对误差**？代码画的是绝对值（`plot.value = TRUE`）。 | 若论文写的是相对误差，则论文与代码不符，需要二选一并在复现说明里写清楚。 |
| U3 | 论文图 7 的 SNR 范围 `0.37 → 3072`（`code-paper-association.md:192`）与本代码 `≈0.375 → ≈768` 的上界不符。 | 我按代码算出的 `num.signal ≡ 9216`、`num.noise ≈ 96λ` 是确定的；上界差异可能来自论文的另一套计数或更早的 λ 网格。**⚠️** |
| U4 | mode 3 的 10 个重复实验共享 `set.seed(0)`（因 `p = 0`）。这是**原意**还是**事故**？ | 决定移植版是否要"修正"为独立种子（会改变图的离散程度）。 |
| U5 | 种子浮点→整数的具体规则（向零截断 vs 四舍五入）未在本机验证（无 R）。 | 只影响"哪两次实验完全相同"的细节，不影响统计结论。 |
| U6 | `rgl` 在本机/CI 上能否加载未验证；`dev.copy(pdf, ...)` 在纯 headless 下是否产出空白页未验证。 | 只影响 mode 4/5/8（图 6）。 |
| U7 | 96×96 规模上 `hybrid.optim` 的实际外层迭代次数（我在 24×24 代理数据上实测 2–7）。 | 只影响成本估算的常数因子，不影响量级。 |

---

## 附录 A：验证方法与证据

本机**没有 R**（`which R`/`Rscript` 均不存在），也**没有 numpy/pandas/scipy**，因此所有数值结论都用纯 Python（`math` + list）重实现后验证，脚本是临时验证脚本（**未纳入仓库**，因为本任务只要求产出规范文档；做法与全部数值结果写在下面，任何有 numpy 的环境都能在几分钟内独立复算）：

1. **解析梯度/海森阵的正确性（✅）**：在 9×9 网格、含随机缺失格（`ts = 0`）的数据上，把 `gra`(`:380-500`) 与中心差分梯度（步长 `1e-5`）对比，最大相对偏差 `7.0e-10`；`hes`(`:415-515`) 与差分海森阵（步长 `1e-4`）对比，最大相对偏差 `2.3e-7`。作为对照，同一组解析式与 `cone.model.inverted` 的差分梯度/海森阵偏差为 `9.0` / `2.34` ⇒ **`gra/hes` 是 `cone.model`（速度版）的精确导数，不是 `cone.model.inverted` 的**。这同时排除了"`dx0x0` 写错了"的怀疑（其形式与教科书写法不同但恒等）。
2. **`grid.optim` 的迭代次数（✅）**：实现 `:556-606` 并在计数器中累加 callnack 次数，得到每次 `grid.optim` 恰好 **62 轮 × 4 次 = 248 次**评估，与"半宽 1000、每轮 ×2/3、`ε = 1e-8`"的解析计算 `n = 62` 一致，且与数据无关。
3. **`hybrid.optim` 的收敛（✅）**：在 16×16 与 24×24 代理数据上实现 `:631-650`，外层迭代 2–7 次（低噪声 2–3 次，`sigma=1, p=0.8` 时 7 次），估计量收敛到真值（低噪声下 `x0` 误差 < 1e-6）。用于 §4.2 的成本估算与外层次数上界。

未能验证的：任何依赖 R 语义的细节（`set.seed` 的取整、`lm` 的秩亏行为、`sample` 的 RNG 消耗、`hash` 的键序、`rgl` 的可加载性）。这些都在正文里以 **⚠️** 标出。

## 附录 B：给移植者的最小实现清单

按依赖顺序实现（活跃路径）：

1. `simulate_circular_2d(theta, limit=96, time.max=96, ts=2, x=48, y=48, miux=miuy=1, gap=1e6, ...) -> ndarray (n,4)`（对应 `stomach.sim2d` + `stomach.sim.one2d`，`:100-198`；RNG 顺序见 §1.3，种子公式见 §1.2，并**显式记录**你选择的种子策略）。
2. `cone_model_inverted_all(pp, sim)`（`:316-325`）——**把它向量化**。
3. `grid_find_best_point_1d` / `grid_find_best_point` / `grid_optim`（`:556-606`）——注意 T15 的拷贝语义、T16 的每维 2 点。
4. `vt_optim_all`（`:620-629`）——`np.polyfit`/最小二乘，注意"先 t0（截距）后 v（斜率）"与"第 3 列是因变量"。
5. `hybrid_optim`（`:631-650`）——交替，`±500` 初界，`ε = 1e-8`，上限 10000。
6. `visualize2d`（`:652-730`）——返回 `truth = (48,48,1,2)` 与 7 维 `pp`。
7. `optimality_table` 的 mode 1/2/3 扫描（`:884-943`）+ `relerror_plot`（`:782-876`，含 8 列约定、`plot.value=True` 的绝对估计语义、三种 x 轴规则）。
8. （可选）3D 图 6 用 matplotlib 替代 `rgl`（`:676-727`）。

**不需要**实现的：`hough`/`point.angle.to.line`/`visualize`/`compute.theta`/`newton.cone*`/`gra`/`hes`/`norm.xy`/`convert.to.grid`/`cone.model*`（网格版）/`stomach.sim*`（1D 版）——它们要么是死代码，要么只服务文件里已被注释的分支。若要为"算法完整性"实现 `gra/hes`，请照 §2.3/§2.4 的公式（已验证正确），而不要试图从 `cone.model.inverted` 重新推一遍。
