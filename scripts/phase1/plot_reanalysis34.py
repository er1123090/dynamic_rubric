"""RQ1-style operational panels; never substitute step for missing KL."""

import argparse
from collections import defaultdict
import csv
from pathlib import Path
import statistics
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def configure_korean_font(output_dir):
    """Use a Korean-capable font when one is available in the runtime."""
    bundled_font = output_dir / "NotoSansCJKkr-Regular.otf"
    if bundled_font.exists():
        font_manager.fontManager.addfont(bundled_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=bundled_font).get_name()
        plt.rcParams["axes.unicode_minus"] = False
        return
    preferred_families = (
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for family in preferred_families:
        if family in installed:
            plt.rcParams["font.family"] = family
            break
    plt.rcParams["axes.unicode_minus"] = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    data = {}
    for side in ("inference_b", "trainer"):
        grouped = defaultdict(list)
        with (args.analysis / f"per_group_{side}.csv").open() as f:
            for row in csv.DictReader(f):
                grouped[int(row["global_step"])].append(row)
        data[side] = grouped
    steps = sorted(data["inference_b"])

    def values(side, key, scale=1):
        return [
            scale * statistics.mean(float(r[key]) for r in data[side][step] if r[key] != "")
            for step in steps
        ]

    configure_korean_font(args.output)
    plt.rcParams.update(
        {
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(
        steps,
        values("inference_b", "stale_zar", 100),
        "s--",
        label="같은 질문의 이전 채점 기준 (stale)",
    )
    ax[0, 0].plot(
        steps,
        values("inference_b", "fresh_zar", 100),
        "o-",
        label="이번에 새로 만든 채점 기준 (fresh)",
    )
    ax[0, 0].set(
        title="A. 16개 응답이 모두 같은 점수를 받은 질문 비율\n"
        "(Exact ZAR; 낮을수록 완전 동점 질문이 적음)",
        ylabel="해당 질문의 비율 (%)",
    )
    for side, marker, label in (
        ("inference_b", "o-", "학습 점수 + Inference B 이전 기준 채점"),
        ("trainer", "s--", "Trainer에서 양쪽 기준 모두 재채점"),
    ):
        ax[0, 1].plot(steps, values(side, "v_adj_zar", 100), marker, label=label)
        ax[1, 0].plot(steps, values(side, "kendall_tau_b"), marker, label=label)
        ax[1, 1].plot(steps, values(side, "delta_separation_rate", 100), marker, label=label)
    ax[0, 1].set(
        title="B. 이번 기준을 쓰면 완전 동점 질문이 얼마나 줄어드는가?\n"
        "(이전 ZAR - 새 ZAR; +이면 개선)",
        ylabel="개선 폭 (%p)",
    )
    ax[1, 0].set(
        title="C. 이전 기준과 이번 기준이 매긴 응답 순서는 비슷한가?\n(Kendall tau-b; 정확도를 뜻하지 않음)",
        ylabel="순위 유사도 (tau-b)",
        ylim=(-0.05, 1.05),
    )
    ax[1, 1].set(
        title="D. 이번 기준은 응답 쌍을 얼마나 더 구분하는가?\n"
        "(새 구분률 - 이전 구분률; 동점 허용폭 0.01, +이면 개선)",
        ylabel="구분률 변화 (%p)",
    )
    for a in (ax[0, 1], ax[1, 1]):
        a.axhline(0, color="gray", linewidth=0.7)
    for a in ax.flat:
        a.set_xlabel("학습 업데이트 번호 (업데이트 직전 모델의 응답을 평가)")
        a.legend(fontsize=8)
        a.set_xticks([17, 20, 24, 28, 32, 34])
    fig.suptitle(
        "그림 4 | 같은 응답을 이전 기준과 이번 기준으로 채점하면 무엇이 달라지는가?\n"
        "실제 학습 응답 비교 · 매번 질문 구성이 다름 · 독립 검증용 Pool B 결과가 아님 · Medicine / OnlineRubrics / seed 11",
        fontsize=12,
    )
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output / f"rq1_style_fresh_stale.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
