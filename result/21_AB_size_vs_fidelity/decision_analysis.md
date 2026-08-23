# Fidelity 调度器决策拆解（exp 21 / Pass B）

对 `21_AB_size_vs_fidelity` 的 **Pass B（policy=fidelity）**运行中的决策现场做拆解。
数据来自实时 `/programs` 快照（t≈2967s，λ=2.30e-8）+ `tape_A.jsonl`（A 的真实 decode）+ `ta_B.log`。

## 设定与公式

- policy = **fidelity**（dual_descent + 峰值预留 + warm/cold + 主动驱逐），`alpha=0.03`
- **价格 λ ≈ 2.30e-8**（分析时刻）
- `prefill 长度 ≈ total_tokens`（当前占用的 KV / 上下文）
- **`decode 预估` = `known_decode` = A（Pass A）在 `(iid, turn)` 的真实 decode 长度**（经 X-Decode-Len 喂入）；tape 里没有该 (iid,turn) 时才回落 `decode_hat=1000`。本快照命中 **14/30 (47%)**。
- **density** = `1 / ((0.03·q + decode)·q)`，`q=total_tokens`
  - ⚠️ 由 **footprint 和 decode 共同决定**，不是单看 footprint
- **准入门**：`_greedy_resume` 只放 `density > λ`，按 density 从高到低
- **驱逐序**：`_pause_until_safe` / 主动驱逐取 `argmin(density)`

## 🔑 核心洞察：decode 让"谁贵"翻盘

用 A 的真 decode 后，density 排序**不再等于 footprint 序**：

| 程序 | prefill | A-decode | density | 说明 |
|---|---:|---:|---:|---|
| sympy-14711 | **14514（小）** | **8192** | 7.99e-9（**最低**） | 小上下文，但 A 说它要 decode 8192 → **最贵 → 头号驱逐** |
| sympy-16450 | 28944（大） | 2695 | 9.70e-9 | 大上下文但 decode 中等 |
| pytest-7982 | 18808 | **329（小）** | 5.95e-8（高） | decode 极短 → 便宜 → **优先准入** |

> **这就是 known_decode oracle 的意义**：footprint 只看"占多大"，decode 才看"要跑多久"。一个 15k 上下文、但要 decode 8k 的程序，比一个 29k 上下文、只 decode 2k 的程序**更贵**。若像旧版 md 那样把 decode 当常数 1000，就会漏掉这一层，排序完全错。

---

## 场景一：有空间 → 准入（实时等待队列，λ=2.30e-8）

`_greedy_resume`：按 density 从高到低，只放 `density > λ`。20 个等待任务：

| 等待程序 | prefill | step | **A-decode** | density | 门槛 |
|---|---:|---:|---:|---:|---|
| **sympy-15349** | 10590 | 6 | 1000 (hat) | 7.17e-8 | **PASS → 首个 admit** ✅ |
| pytest-7982 | 18808 | 9 | **329 (A)** | 5.95e-8 | PASS（decode 极短 → 跳到第 2） |
| sympy-13852 | 13090 | 8 | 1000 (hat) | 5.49e-8 | PASS |
| pytest-7521 | 18322 | 6 | 552 (A) | 4.95e-8 | PASS |
| …（共 **9** 个 PASS） | | | | | |
| — 门槛线 λ=2.30e-8 — | | | | | |
| psf-6028 | 25193 | 14 | 1000 (hat) | 2.26e-8 | **BLOCKED** ❌ |
| pytest-10051 | 36454 | 21 | 1000 (hat) | 1.31e-8 | BLOCKED（footprint 最大） |
| pytest-5262 | 30833 | 17 | 2475 (A) | 9.54e-9 | BLOCKED |
| **sympy-15976** | **16380（小）** | 7 | **8192 (A)** | 7.03e-9 | **BLOCKED**（小上下文，但 decode 8192） |
| **pylint-6528** | 20746 | 9 | **8192 (A)** | 5.47e-9 | **BLOCKED（最低）** ❌ |

**最终选择：admit `sympy-15349`（density 最高）。** 共 9 个 PASS / 11 个 BLOCKED。
注意被挡在最底的 **sympy-15976 / pylint-6528**：footprint 只有 16k/20k（不算大),但 A-decode=**8192** → density 垫底 → 被价格门挡住。**纯看 footprint 会把它们放进来，正是 decode oracle 把它们拦下。**

---

## 场景二：KV 爆了 → 驱逐

### (a) 当前 active 集合的驱逐候选（同一快照）

`argmin(density)`（含 decode）→ 3 个 active 已跌破 λ：

| active 程序 | prefill | step | **A-decode** | density | |
|---|---:|---:|---:|---:|---|
| **sympy-14711** | 14514 | 6 | **8192 (A)** | 7.99e-9 | **<λ → 头号驱逐候选**（小 footprint 大 decode） |
| sympy-16450 | 28944 | 9 | 2695 (A) | 9.70e-9 | <λ |
| pytest-5787 | 21623 | 10 | 1993 (A) | 1.75e-8 | <λ |
| sympy-12489 | 22697 | 17 | 1120 (A) | 2.45e-8 | >λ（保留） |
| …psf-2317 | 12725 | 9 | 1000 (hat) | 5.69e-8 | >λ（最高，最该留） |

> 若纯按 footprint，头号受害者会是 sympy-16450（28944 最大）；**用真 decode 后是 sympy-14711**（14514 小，但要 decode 8192）。decode 改变了驱逐对象。

### (b) 真实驱逐事件（`ta_B.log:5571–5622`，KV 溢出）

`remaining_capacity<0` → `_pause_until_safe` 标记 10 个 REASONING 待暂停（按 argmin density）。**非抢占**：标记后要等 tool 边界才真 pause，其间还在膨胀：

| 程序 | 标记时 tok | 实际暂停时 tok | 增长 |
|---|---:|---:|---:|
| sympy-12489 | 23357 | 34339 | **+47%** |
| sympy-13878 | 24434 | 29181 | +19% |

> 一轮内从 23k 涨到 34k —— 这正是 preemption 的来源、也是 fidelity **Gap 1（峰值预留）** 想解决的：准入时若只按当前 footprint 算就会撑爆。

---

## 合起来：机制自洽

同一价格线 **λ=2.30e-8** 管两端，且 density **= f(footprint, decode)**：

- **准入**挑 density 最高：sympy-15349 / pytest-7982（footprint 中等但 decode 短 → 便宜）
- **驱逐**挑 density 最低：sympy-14711 / sympy-15976 / pylint-6528（footprint 未必大，但 **A-decode=8192** → 贵）
- decode oracle 让"小上下文但会长 decode"的程序被正确识别为昂贵，是纯 footprint 策略看不到的一层

## caveat

- `decode` 用 A（Pass A）在 `(iid, turn)` 的真实 decode（tape_A）；命中 47%（本快照，深轮次多、A 未必录到），miss 回落 `decode_hat=1000`。
- ⚠️ 这些 A-decode 值本身**大多是过期/不准的**：temp=0 在 vLLM 下不确定，A 与 B 的同 (iid,turn) decode 仅约 3% 一致（batch-variance 漂移）。所以喂进去的"已知 decode"是 A 的值、不等于 B 真实会 decode 的量。要让 oracle 真准，需 batch-invariant vLLM 或 replay 锁轨迹。
- density/warm-cold 为近似（未含 warm 折扣）；但结论（decode 主导小-footprint-大-decode 程序的排序）稳健。
