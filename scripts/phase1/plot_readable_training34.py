"""Relabel the published training figures from unchanged saved CSVs; no inference."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def read_rows(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--font", type=Path, required=True)
    args = parser.parse_args()
    font_manager.fontManager.addfont(str(args.font))
    plt.rcParams.update(
        {
            "font.family": font_manager.FontProperties(fname=str(args.font)).get_name(),
            "font.size": 11,
            "svg.fonttype": "path",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    inputs = [
        args.training / name for name in ("training_by_step.csv", "criterion_pooled_by_step.csv")
    ]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    steps = read_rows(inputs[0])
    pooled = [r for r in read_rows(inputs[1]) if r["component"] == "fresh"]
    x = [int(r["step"]) for r in steps]
    assert x == [int(r["step"]) for r in pooled]

    def y(key, scale=1):
        return [scale * float(r[key]) for r in steps]

    args.output.mkdir(parents=True, exist_ok=True)
    evidence = {}

    def save(fig, axes, name):
        for axis in axes:
            axis.set_xlabel("학습 업데이트 번호 (응답은 업데이트 직전 모델에서 생성)")
            axis.tick_params(labelsize=10)
            axis.legend(fontsize=9)
        fig.tight_layout(rect=(0, 0.03, 1, 0.91), w_pad=2.4)
        evidence[name] = [
            [
                {
                    "x": line.get_xdata().tolist()
                    if hasattr(line.get_xdata(), "tolist")
                    else list(line.get_xdata()),
                    "y": line.get_ydata().tolist()
                    if hasattr(line.get_ydata(), "tolist")
                    else list(line.get_ydata()),
                    "label": line.get_label(),
                }
                for line in axis.lines
            ]
            for axis in axes
        ]
        for suffix in ("png", "svg"):
            fig.savefig(args.output / f"{name}.{suffix}", dpi=170, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.1))
    for key, marker, label in (
        ("reward_mean", "o-", "평균 점수"),
        ("reward_one_fraction", "s--", "만점(1점) 응답 비중"),
        ("reward_zero_fraction", "^--", "0점 응답 비중"),
    ):
        ax[0].plot(x, y(key), marker, label=label)
    ax[0].set(
        title="A. 평균 점수와 0점·만점 응답은 얼마나 많은가?",
        ylabel="점수 또는 비중 (0.1 = 응답의 10%)",
        ylim=(0, 1),
    )
    ax[1].plot(x, y("fresh_exact_zero_advantage", 100), "o-", label="16개 모두 동점 (Exact ZAR)")
    ax[1].plot(
        x, y("fresh_near_zero_advantage", 100), "s--", label="거의 동점 (점수 표준편차 ≤ 0.01)"
    )
    ax[1].set(title="B. 응답 16개에 점수 차이가 없는 질문은?", ylabel="해당 질문의 비율 (%)")
    for kind, label in (
        ("effective", "일부 응답만 충족 (응답 구분에 참여)"),
        ("saturated", "16개 응답 모두 충족"),
        ("dead", "16개 응답 모두 미충족"),
    ):
        ax[2].plot(x, [100 * float(r[kind + "_ratio"]) for r in pooled], "o-", label=label)
    ax[2].set(title="C. 채점 항목이 응답을 어떻게 구분하는가?", ylabel="전체 채점 항목 중 비율 (%)")
    fig.suptitle("그림 1 | 학습 중 점수와 채점 항목의 변화", fontsize=17, y=0.99)
    fig.text(
        0.5,
        0.90,
        "매 방문 새로 만든 기준으로 실제 학습 응답을 채점 · 업데이트 1–34 · Medicine / OnlineRubrics / seed 11",
        ha="center",
        fontsize=11,
    )
    save(fig, ax, "training_readable")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.1))
    ax[0].plot(x, y("candidate_count"), label="중복 제거 전 후보")
    ax[0].plot(x, y("online_count"), label="중복 제거 후 추가 항목")
    ax[0].set(title="A. 새 채점 항목은 몇 개 남는가?", ylabel="질문당 평균 항목 수")
    for prefix, label, color in (
        ("offline_component", "초기 항목만 (R0)", "#0072b2"),
        ("online", "이번에 추가한 항목만", "#d55e00"),
    ):
        ax[1].plot(x, y(prefix + "_effective_criterion_ratio", 100), label=label, color=color)
    ax[1].set(
        title="B. 일부 응답만 충족하는 항목은 얼마나 되는가?", ylabel="질문별 비율의 평균 (%)"
    )
    ax[2].plot(x, y("offline_component_pairwise_tie_rate", 100), label="초기 항목만 점수에 반영")
    ax[2].plot(x, y("fresh_pairwise_tie_rate", 100), label="초기 + 추가 항목 모두 반영")
    ax[2].set(
        title="C. 추가 항목을 반영하면 동점 쌍이 줄어드는가?",
        ylabel="점수 차이 ≤ 0.01인 응답 쌍 (%)",
    )
    for axis in ax:
        axis.axvline(16.5, color="gray", ls=":")
    fig.suptitle("보조 그림 | 새 채점 항목이 응답 구분에 기여하는 방식", fontsize=17, y=0.99)
    fig.text(
        0.5,
        0.90,
        "전체 기준으로 채점한 결과를 항목별로 나눠 집계 · 과거 기준과의 비교나 R0 별도 재채점이 아님",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.01,
        "세로 점선: 첫 번째 학습 데이터 순회 종료 (업데이트 16 이후)",
        ha="center",
        fontsize=10,
    )
    save(fig, ax, "criteria_readable")
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output / "training_plot_series.json").write_text(
        json.dumps({"input_sha256": hashes, "series": evidence}, ensure_ascii=False, indent=2)
        + "\n"
    )
    print(
        json.dumps(
            {"inputs_unchanged": True, "steps": len(x), "figures": list(evidence)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
