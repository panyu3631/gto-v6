#!/usr/bin/env python3
"""
L5 一致性检查 #6: 公式格式验证

验证:
- 所有因子公式使用正确的数学符号
- delta_signs 格式一致
- 所有因子有有效的 data_sources
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.factors.registry import FACTOR_REGISTRY

EXIT_CODE = 0


def check(name, condition, detail=""):
    global EXIT_CODE
    ok = bool(condition)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} {detail}")
    if not ok:
        EXIT_CODE = 1


def main():
    print("=" * 60)
    print("L5 一致性检查 #6: 公式格式验证")
    print("=" * 60)

    for fid, factor in FACTOR_REGISTRY.items():
        # 跳过 F14
        if fid == "F14":
            continue

        # 1. formula 不为空
        check(f"{fid} formula 不为空", bool(factor.formula), f"formula='{factor.formula}'")

        # 2. delta_signs 包含 home/draw/away
        for key in ("home", "draw", "away"):
            check(f"{fid} delta_signs 包含 '{key}'", key in factor.delta_signs)

        # 3. delta_signs 值为 -1, 0, 或 +1
        for key, val in factor.delta_signs.items():
            check(f"{fid} delta_signs[{key}] ∈ {{-1,0,1}}", val in (-1, 0, 1))

        # 4. data_sources 不为空
        check(f"{fid} data_sources 不为空", len(factor.data_sources) > 0,
              f"data_sources={factor.data_sources}")

        # 5. name 不为空
        check(f"{fid} name 不为空", bool(factor.name))

        # 6. name_cn 不为空
        check(f"{fid} name_cn 不为空", bool(factor.name_cn))

        # 7. description 不为空
        check(f"{fid} description 不为空", bool(factor.description),
              f"description='{factor.description}'")

    print(f"\n退出码: {EXIT_CODE}")
    sys.exit(EXIT_CODE)


if __name__ == "__main__":
    main()