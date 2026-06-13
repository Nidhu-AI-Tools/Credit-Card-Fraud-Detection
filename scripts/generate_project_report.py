#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import os
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
MPLCONFIGDIR = REPO_ROOT / ".mplconfig"
XDG_CACHE_HOME = REPO_ROOT / ".cache"
MPLCONFIGDIR.mkdir(exist_ok=True)
XDG_CACHE_HOME.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


OUTPUTS_DIR = REPO_ROOT / "outputs"
PREPROCESSING_DIR = OUTPUTS_DIR / "preprocessing"
TRAIN_DIR = OUTPUTS_DIR / "train"
ADVANCED_DIR = OUTPUTS_DIR / "advanced_models"
EVALUATE_DIR = OUTPUTS_DIR / "evaluate"

PAGE_SIZE = (11.69, 8.27)
PAGE_LEFT = 0.06
PAGE_RIGHT = 0.95
PAGE_TOP = 0.93
BODY_TOP = 0.84
BODY_BOTTOM = 0.08

COLOR_INK = "#13293d"
COLOR_MUTED = "#5b6b79"
COLOR_ACCENT = "#0b5d7a"
COLOR_ACCENT_LIGHT = "#d9eef6"
COLOR_BG = "#f7f9fb"
COLOR_SUCCESS = "#1f7a5c"
COLOR_WARN = "#b45309"


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    unnamed = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed:
        df = df.rename(columns={unnamed[0]: "Metric"})
    return df


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text().strip()


def format_float(value: float | int, digits: int = 4) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def chunked(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def wrap_value(value: object, width: int = 36) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width))


def clean_table(df: pd.DataFrame, max_rows: int = 12, wrap_width: int = 36) -> pd.DataFrame:
    preview = df.head(max_rows).copy()
    for col in preview.columns:
        preview[col] = preview[col].map(lambda value: wrap_value(value, width=wrap_width))
    return preview


def create_figure() -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    return fig


def save_figure(pdf: PdfPages, fig: plt.Figure) -> None:
    pdf.savefig(fig, facecolor=fig.get_facecolor())
    plt.close(fig)


def add_text_page(pdf: PdfPages, title: str, lines: Iterable[str], footer: str | None = None) -> None:
    wrapped_blocks: list[list[str]] = []
    for line in lines:
        if not str(line).strip():
            wrapped_blocks.append([""])
            continue
        wrapped_blocks.append(textwrap.wrap(str(line), width=105) or [""])

    pages: list[list[list[str]]] = []
    current_page: list[list[str]] = []
    current_height = 0.0
    page_capacity = 13.5

    for block in wrapped_blocks:
        block_height = max(1.0, len(block) * 0.95) + 0.35
        if current_page and current_height + block_height > page_capacity:
            pages.append(current_page)
            current_page = []
            current_height = 0.0
        current_page.append(block)
        current_height += block_height

    if current_page:
        pages.append(current_page)

    if not pages:
        pages = [[["No content available."]]]

    for page_number, page_blocks in enumerate(pages, start=1):
        fig = create_figure()
        page_title = title if len(pages) == 1 else f"{title} ({page_number}/{len(pages)})"
        fig.text(PAGE_LEFT, PAGE_TOP, page_title, fontsize=22, fontweight="bold", ha="left", va="top")

        y = BODY_TOP
        for block in page_blocks:
            if block == [""]:
                y -= 0.03
                continue

            fig.text(PAGE_LEFT + 0.01, y, block[0], fontsize=11.5, ha="left", va="top")
            y -= 0.038
            for continuation in block[1:]:
                fig.text(PAGE_LEFT + 0.03, y, continuation, fontsize=11.5, ha="left", va="top")
                y -= 0.034
            y -= 0.016

        if footer:
            fig.text(PAGE_LEFT, 0.04, footer, fontsize=9, color="#555555", ha="left")
        fig.text(PAGE_RIGHT, 0.04, f"Page {page_number} of {len(pages)}", fontsize=9, color="#555555", ha="right")

        save_figure(pdf, fig)


def add_table_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    subtitle: str | None = None,
    max_rows: int = 12,
    font_size: int = 10,
    scale_y: float = 1.5,
) -> None:
    row_chunks = chunked(list(range(len(df))), max_rows)
    if not row_chunks:
        row_chunks = [[]]

    wrap_width = max(14, min(34, int(130 / max(1, len(df.columns)))))

    for page_number, row_idx_chunk in enumerate(row_chunks, start=1):
        fig = create_figure()
        page_title = title if len(row_chunks) == 1 else f"{title} ({page_number}/{len(row_chunks)})"
        fig.text(PAGE_LEFT, PAGE_TOP, page_title, fontsize=20, fontweight="bold", ha="left", va="top")
        if subtitle:
            fig.text(PAGE_LEFT, 0.885, subtitle, fontsize=11, ha="left", va="top")

        ax = fig.add_axes([PAGE_LEFT, 0.11, PAGE_RIGHT - PAGE_LEFT, 0.68])
        ax.axis("off")

        if row_idx_chunk:
            table_df = clean_table(df.iloc[row_idx_chunk], max_rows=max_rows, wrap_width=wrap_width)
        else:
            table_df = pd.DataFrame({"Status": ["No data available"]})

        dynamic_font_size = max(7.5, font_size - max(0, len(table_df.columns) - 5) * 0.35)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(dynamic_font_size)
        table.scale(1, scale_y)

        try:
            table.auto_set_column_width(col=list(range(len(table_df.columns))))
        except Exception:
            pass

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("#0b5d7a")
            else:
                cell.set_facecolor("#f7f9fb" if row % 2 == 0 else "white")
                if col == 0:
                    cell._loc = "left"

        save_figure(pdf, fig)


def add_image_grid_page(pdf: PdfPages, title: str, items: list[tuple[Path, str]]) -> None:
    existing = [(path, caption) for path, caption in items if path.exists()]
    if not existing:
        add_text_page(pdf, title, ["No saved image artifacts were found for this section."])
        return

    for page_number, chunk in enumerate(chunked(existing, 2), start=1):
        fig = create_figure()
        page_title = title if len(existing) <= 2 else f"{title} ({page_number}/{int(np.ceil(len(existing) / 2))})"
        fig.text(PAGE_LEFT, PAGE_TOP, page_title, fontsize=20, fontweight="bold", ha="left", va="top")

        if len(chunk) == 1:
            axes = [fig.add_axes([0.09, 0.16, 0.82, 0.62])]
        else:
            axes = [
                fig.add_axes([0.09, 0.54, 0.82, 0.24]),
                fig.add_axes([0.09, 0.16, 0.82, 0.24]),
            ]

        for ax, (path, caption) in zip(axes, chunk):
            ax.imshow(mpimg.imread(path))
            ax.axis("off")
            ax.set_title(caption, fontsize=12, pad=10)

        save_figure(pdf, fig)


def add_metric_bar_page(
    pdf: PdfPages,
    title: str,
    df: pd.DataFrame,
    index_col: str,
    metric_cols: list[str],
    subtitle: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    plot_df = df.set_index(index_col)[metric_cols]
    plot_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_title(title, fontsize=20, fontweight="bold", loc="left", pad=18)
    if subtitle:
        fig.text(0.06, 0.91, subtitle, fontsize=11, ha="left")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, plot_df.max().max() * 1.1))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(rect=(0.03, 0.03, 0.98, 0.90))
    save_figure(pdf, fig)


def add_threshold_tradeoff_page(pdf: PdfPages, title: str, df: pd.DataFrame, label_prefix: str = "") -> None:
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax.plot(df["Threshold"], df["Precision"], marker="o", label=f"{label_prefix}Precision".strip())
    ax.plot(df["Threshold"], df["Recall"], marker="o", label=f"{label_prefix}Recall".strip())
    ax.plot(df["Threshold"], df["F1 Score"], marker="o", label=f"{label_prefix}F1".strip())
    ax.set_title(title, fontsize=20, fontweight="bold", loc="left", pad=18)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend()
    best_idx = df["F1 Score"].idxmax()
    best_row = df.loc[best_idx]
    ax.axvline(best_row["Threshold"], color="#aa3333", linestyle="--", alpha=0.6)
    ax.text(
        best_row["Threshold"],
        best_row["F1 Score"] + 0.02,
        f"Best F1 @ {best_row['Threshold']}",
        color="#aa3333",
        fontsize=10,
        ha="center",
    )
    plt.tight_layout(rect=(0.03, 0.03, 0.98, 0.94))
    save_figure(pdf, fig)


def add_weighted_comparison_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=20, fontweight="bold", x=0.06, ha="left")

    metric_pairs = [
        ("Unweighted Precision", "Weighted Precision", "Precision"),
        ("Unweighted Recall", "Weighted Recall", "Recall"),
        ("Unweighted F1", "Weighted F1", "F1 Score"),
    ]

    for ax, (unweighted_col, weighted_col, label) in zip(axes, metric_pairs):
        ax.plot(df["Threshold"], df[unweighted_col], marker="o", label="Unweighted")
        ax.plot(df["Threshold"], df[weighted_col], marker="o", label="Weighted")
        ax.set_title(label)
        ax.set_xlabel("Threshold")
        ax.grid(alpha=0.25)
        ax.legend()

    plt.tight_layout(rect=(0.03, 0.03, 0.98, 0.92))
    save_figure(pdf, fig)


def add_confusion_matrix_page(pdf: PdfPages, df: pd.DataFrame) -> None:
    rows = int(np.ceil(len(df) / 2))
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    fig.suptitle("Finalist Confusion Matrices", fontsize=20, fontweight="bold", x=0.06, ha="left")
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    vmax = df[["TN", "FP", "FN", "TP"]].to_numpy().max()

    for ax in axes_array.flat:
        ax.axis("off")

    for ax, (_, row) in zip(axes_array.flat, df.iterrows()):
        matrix = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]], dtype=float)
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title(f"{row['Model']}\nThreshold = {row['Threshold']}", fontsize=10)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="#111111", fontsize=11)

    plt.tight_layout(rect=(0.03, 0.03, 0.98, 0.92))
    save_figure(pdf, fig)


def add_feature_importance_page(pdf: PdfPages, df: pd.DataFrame, title: str) -> None:
    plot_df = df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax.barh(plot_df["Feature"], plot_df["Importance"], color="#0b5d7a")
    ax.set_title(title, fontsize=20, fontweight="bold", loc="left", pad=18)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout(rect=(0.03, 0.03, 0.98, 0.94))
    save_figure(pdf, fig)


def add_cover_page(pdf: PdfPages, best_model_row: pd.Series | None, generated_at: str) -> None:
    fig = create_figure()
    fig.patch.set_facecolor(COLOR_BG)
    fig.add_artist(plt.Rectangle((0, 0), 1, 1, transform=fig.transFigure, color=COLOR_BG))
    fig.add_artist(plt.Rectangle((0, 0.84), 1, 0.16, transform=fig.transFigure, color=COLOR_INK))
    fig.add_artist(plt.Rectangle((0.06, 0.16), 0.88, 0.03, transform=fig.transFigure, color=COLOR_ACCENT))

    fig.text(0.06, 0.92, "Credit Card Fraud Detection", fontsize=28, fontweight="bold", color="white", ha="left", va="top")
    fig.text(0.06, 0.87, "Project Report", fontsize=16, color="#dce6ee", ha="left", va="top")

    fig.text(0.06, 0.73, "Objective", fontsize=13, fontweight="bold", color=COLOR_ACCENT, ha="left")
    fig.text(
        0.06,
        0.68,
        "Build a fraud-detection workflow from data understanding through final model selection,\nthen evaluate the chosen model once on an untouched test set.",
        fontsize=14,
        color=COLOR_INK,
        ha="left",
        va="top",
    )

    fig.text(0.06, 0.52, "Final Outcome", fontsize=13, fontweight="bold", color=COLOR_ACCENT, ha="left")
    if best_model_row is not None:
        fig.text(0.06, 0.47, str(best_model_row["Model"]), fontsize=24, fontweight="bold", color=COLOR_INK, ha="left")
        card_x = [0.06, 0.28, 0.50, 0.72]
        labels = [
            ("PR-AUC", format_float(best_model_row["PR-AUC"])),
            ("Precision", format_float(best_model_row["Precision"])),
            ("Recall", format_float(best_model_row["Recall"])),
            ("F1", format_float(best_model_row["F1 Score"])),
        ]
        for x, (label, value) in zip(card_x, labels):
            fig.add_artist(plt.Rectangle((x, 0.30), 0.18, 0.11, transform=fig.transFigure, color="white", ec="#d0d7de"))
            fig.text(x + 0.02, 0.38, label, fontsize=11, color=COLOR_MUTED, ha="left")
            fig.text(x + 0.02, 0.33, value, fontsize=18, fontweight="bold", color=COLOR_INK, ha="left")

    fig.text(0.06, 0.10, f"Generated: {generated_at}", fontsize=10, color=COLOR_MUTED, ha="left")
    fig.text(0.94, 0.10, "Source notebooks: preprocessing, train, advanced_models, evaluate", fontsize=10, color=COLOR_MUTED, ha="right")
    save_figure(pdf, fig)


def add_section_divider(pdf: PdfPages, section_number: str, title: str, subtitle: str) -> None:
    fig = create_figure()
    fig.patch.set_facecolor(COLOR_INK)
    fig.add_artist(plt.Rectangle((0.06, 0.18), 0.88, 0.64, transform=fig.transFigure, color="#16354f"))
    fig.text(0.10, 0.70, section_number, fontsize=14, fontweight="bold", color="#8dd3ea", ha="left")
    fig.text(0.10, 0.56, title, fontsize=28, fontweight="bold", color="white", ha="left")
    fig.text(0.10, 0.44, subtitle, fontsize=14, color="#dce6ee", ha="left", va="top")
    save_figure(pdf, fig)


def add_bullets_and_table_page(
    pdf: PdfPages,
    title: str,
    bullets: list[str],
    df: pd.DataFrame,
    table_title: str,
    subtitle: str | None = None,
    max_rows: int = 8,
) -> None:
    fig = create_figure()
    fig.text(PAGE_LEFT, PAGE_TOP, title, fontsize=21, fontweight="bold", color=COLOR_INK, ha="left", va="top")
    if subtitle:
        fig.text(PAGE_LEFT, 0.885, subtitle, fontsize=11, color=COLOR_MUTED, ha="left", va="top")

    left_ax = fig.add_axes([0.06, 0.14, 0.41, 0.66])
    left_ax.axis("off")
    y = 0.98
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=43)
        left_ax.text(0.0, y, u"\u2022", fontsize=16, color=COLOR_ACCENT, va="top")
        left_ax.text(0.05, y, "\n".join(wrapped), fontsize=11.5, color=COLOR_INK, va="top")
        y -= 0.11 + 0.05 * max(0, len(wrapped) - 1)
        if y < 0.08:
            break

    table_ax = fig.add_axes([0.52, 0.14, 0.42, 0.66])
    table_ax.axis("off")
    table_ax.text(0.0, 1.02, table_title, fontsize=12, fontweight="bold", color=COLOR_ACCENT, va="bottom")
    table_df = clean_table(df, max_rows=max_rows, wrap_width=max(16, min(34, int(120 / max(1, len(df.columns))))))
    table = table_ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc="center", bbox=[0, 0, 1, 0.95])
    table.auto_set_font_size(False)
    table.set_fontsize(max(8, 10 - max(0, len(table_df.columns) - 5) * 0.3))
    try:
        table.auto_set_column_width(col=list(range(len(table_df.columns))))
    except Exception:
        pass
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d7de")
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(COLOR_ACCENT)
        else:
            cell.set_facecolor("#f7f9fb" if row % 2 == 0 else "white")
    save_figure(pdf, fig)


def add_bullets_and_chart_page(
    pdf: PdfPages,
    title: str,
    bullets: list[str],
    chart_df: pd.DataFrame,
    index_col: str,
    metric_cols: list[str],
    subtitle: str | None = None,
    kind: str = "bar",
) -> None:
    fig = create_figure()
    fig.text(PAGE_LEFT, PAGE_TOP, title, fontsize=21, fontweight="bold", color=COLOR_INK, ha="left", va="top")
    if subtitle:
        fig.text(PAGE_LEFT, 0.885, subtitle, fontsize=11, color=COLOR_MUTED, ha="left", va="top")

    left_ax = fig.add_axes([0.06, 0.16, 0.36, 0.64])
    left_ax.axis("off")
    y = 0.98
    for bullet in bullets:
        wrapped = textwrap.wrap(bullet, width=38)
        left_ax.text(0.0, y, u"\u2022", fontsize=16, color=COLOR_ACCENT, va="top")
        left_ax.text(0.06, y, "\n".join(wrapped), fontsize=11.5, color=COLOR_INK, va="top")
        y -= 0.11 + 0.05 * max(0, len(wrapped) - 1)
        if y < 0.08:
            break

    chart_ax = fig.add_axes([0.48, 0.16, 0.45, 0.64])
    plot_df = chart_df.set_index(index_col)[metric_cols]
    if kind == "barh":
        plot_df.plot(kind="barh", ax=chart_ax, width=0.8, color=["#0b5d7a", "#4c956c", "#b8c0c8", "#e0a458"][: len(metric_cols)])
    else:
        plot_df.plot(kind="bar", ax=chart_ax, width=0.78, color=["#0b5d7a", "#4c956c", "#e0a458", "#8a5cf6"][: len(metric_cols)])
    chart_ax.grid(axis="y" if kind != "barh" else "x", alpha=0.25)
    chart_ax.set_xlabel("")
    chart_ax.set_ylabel("")
    chart_ax.legend(loc="best")
    if kind != "barh":
        chart_ax.tick_params(axis="x", rotation=20)
    save_figure(pdf, fig)


def add_two_images_page(pdf: PdfPages, title: str, items: list[tuple[Path, str]], subtitle: str | None = None) -> None:
    existing = [(path, caption) for path, caption in items if path.exists()]
    if not existing:
        return
    fig = create_figure()
    fig.text(PAGE_LEFT, PAGE_TOP, title, fontsize=21, fontweight="bold", color=COLOR_INK, ha="left", va="top")
    if subtitle:
        fig.text(PAGE_LEFT, 0.885, subtitle, fontsize=11, color=COLOR_MUTED, ha="left", va="top")

    axes = []
    if len(existing) == 1:
        axes = [fig.add_axes([0.10, 0.18, 0.80, 0.58])]
    else:
        axes = [fig.add_axes([0.07, 0.18, 0.40, 0.58]), fig.add_axes([0.53, 0.18, 0.40, 0.58])]

    for ax, (path, caption) in zip(axes, existing[:2]):
        ax.imshow(mpimg.imread(path))
        ax.axis("off")
        ax.set_title(caption, fontsize=12, pad=8)
    save_figure(pdf, fig)


def build_preprocessing_highlights(
    dedup_df: pd.DataFrame | None,
    time_amount_df: pd.DataFrame | None,
    fraud_time_amount_df: pd.DataFrame | None,
    top_bins_df: pd.DataFrame | None,
) -> list[str]:
    lines: list[str] = []
    if dedup_df is not None:
        row = dedup_df.iloc[0]
        lines.append(
            f"After deduplication, the dataset contains {int(row['rows_after_deduplication']):,} transactions with only {int(row['fraud_count'])} fraud cases."
        )
    if time_amount_df is not None and fraud_time_amount_df is not None:
        amount_overall = time_amount_df[time_amount_df["Metric"] == "Amount"].iloc[0]
        amount_fraud = fraud_time_amount_df[fraud_time_amount_df["Metric"] == "Amount"].iloc[0]
        lines.append(
            f"Fraud transactions have a higher mean amount ({amount_fraud['mean']:.2f}) than the overall dataset ({amount_overall['mean']:.2f}), but their median amount remains low."
        )
    if top_bins_df is not None and not top_bins_df.empty:
        top = top_bins_df.iloc[0]
        lines.append(
            f"The highest observed fraud-rate time bin was {top['TimeBin']} with a fraud rate of {top['FraudRate']:.4f}."
        )
    lines.append("The dataset is extremely imbalanced, so ranking quality and the precision-recall tradeoff matter more than plain accuracy.")
    return lines


def build_modeling_highlights(
    baseline_df: pd.DataFrame | None,
    logreg_threshold_df: pd.DataFrame | None,
    finalists_df: pd.DataFrame | None,
) -> list[str]:
    lines: list[str] = []
    if baseline_df is not None:
        baseline_df = baseline_df.rename(columns={baseline_df.columns[0]: "Model"})
        logistic = baseline_df[baseline_df["Model"].str.contains("Logistic", case=False)].iloc[0]
        tree = baseline_df[baseline_df["Model"].str.contains("Decision Tree", case=False)].iloc[0]
        lines.append(
            f"Logistic regression established a strong baseline with PR-AUC {logistic['PR-AUC']:.4f}, while the simple decision tree underperformed on the primary metric."
        )
        lines.append(
            f"The tree model achieved higher recall ({tree['Recall']:.4f}) but at an impractically low precision ({tree['Precision']:.4f})."
        )
    if logreg_threshold_df is not None:
        best = logreg_threshold_df.loc[logreg_threshold_df["F1 Score"].idxmax()]
        lines.append(
            f"The best logistic threshold was {best['Threshold']}, which gave the best middle ground between precision ({best['Precision']:.4f}) and recall ({best['Recall']:.4f})."
        )
    if finalists_df is not None:
        lines.append(
            f"Model exploration narrowed the field to {len(finalists_df)} finalists: " + ", ".join(finalists_df["Model"].tolist()) + "."
        )
    return lines


def derive_best_model_error_summary(final_results_df: pd.DataFrame | None, test_predictions_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if final_results_df is None or test_predictions_df is None or final_results_df.empty:
        return None
    best_model_name = final_results_df.iloc[0]["Model"]
    prefix = best_model_name.lower().replace(" ", "_")
    pred_col = f"{prefix}_pred"
    if pred_col not in test_predictions_df.columns:
        return None

    y_true = test_predictions_df["y_true"]
    y_pred = test_predictions_df[pred_col]
    outcomes = np.select(
        [
            (y_true == 0) & (y_pred == 0),
            (y_true == 0) & (y_pred == 1),
            (y_true == 1) & (y_pred == 0),
            (y_true == 1) & (y_pred == 1),
        ],
        ["True Negative", "False Positive", "False Negative", "True Positive"],
        default="Unknown",
    )
    summary = pd.Series(outcomes).value_counts().rename_axis("Outcome").reset_index(name="Count")
    summary["Proportion"] = summary["Count"] / len(test_predictions_df)
    return summary


def build_final_evaluation_highlights(final_results_df: pd.DataFrame | None, error_summary_df: pd.DataFrame | None) -> list[str]:
    lines: list[str] = []
    if final_results_df is not None:
        best = final_results_df.iloc[0]
        runner_up = final_results_df.iloc[1] if len(final_results_df) > 1 else None
        lines.append(
            f"{best['Model']} is the final winner with PR-AUC {best['PR-AUC']:.4f}, precision {best['Precision']:.4f}, recall {best['Recall']:.4f}, and F1 {best['F1 Score']:.4f}."
        )
        if runner_up is not None:
            lines.append(
                f"The margin over the runner-up ({runner_up['Model']}) is small but real on PR-AUC: {best['PR-AUC'] - runner_up['PR-AUC']:.4f}."
            )
        rf = final_results_df[final_results_df["Model"].str.contains("Random Forest", case=False)]
        if not rf.empty:
            rf_row = rf.iloc[0]
            lines.append(
                f"The best Random Forest variant remained competitive, especially on recall ({rf_row['Recall']:.4f}), but the boosted models ranked higher overall."
            )
    if error_summary_df is not None:
        fp = error_summary_df.loc[error_summary_df["Outcome"] == "False Positive", "Count"]
        fn = error_summary_df.loc[error_summary_df["Outcome"] == "False Negative", "Count"]
        fp_count = int(fp.iloc[0]) if not fp.empty else 0
        fn_count = int(fn.iloc[0]) if not fn.empty else 0
        lines.append(f"The winning model made only {fp_count} false positives and {fn_count} false negatives on the test set.")
    lines.append("The final recommendation is to carry Tuned XGBoost forward, with Tuned LightGBM as the strongest backup candidate.")
    return lines


def build_executive_summary(
    dedup_df: pd.DataFrame | None,
    baseline_df: pd.DataFrame | None,
    final_results_df: pd.DataFrame | None,
) -> list[str]:
    lines: list[str] = []
    if dedup_df is not None:
        row = dedup_df.iloc[0]
        lines.append(
            "The working dataset was deduplicated to "
            f"{int(row['rows_after_deduplication']):,} rows, leaving {int(row['fraud_count'])} fraud cases "
            f"({row['fraud_proportion'] * 100:.3f}% of the data)."
        )
    if baseline_df is not None:
        logistic_row = baseline_df[baseline_df.iloc[:, 0].astype(str).str.contains("Logistic", case=False)].iloc[0]
        lines.append(
            "The strongest training-stage baseline was logistic regression, with "
            f"PR-AUC {logistic_row['PR-AUC']:.4f} and ROC-AUC {logistic_row['ROC-AUC']:.4f}."
        )
    if final_results_df is not None:
        best = final_results_df.iloc[0]
        runner_up = final_results_df.iloc[1] if len(final_results_df) > 1 else None
        lines.append(
            f"The final test winner was {best['Model']} at threshold {best['Threshold']}, "
            f"with PR-AUC {best['PR-AUC']:.4f}, precision {best['Precision']:.4f}, "
            f"recall {best['Recall']:.4f}, and F1 {best['F1 Score']:.4f}."
        )
        if runner_up is not None:
            lines.append(
                f"The closest backup was {runner_up['Model']}, trailing by "
                f"{best['PR-AUC'] - runner_up['PR-AUC']:.4f} PR-AUC points."
            )
    lines.append(
        "The overall project conclusion is that tuned boosting models clearly outperform the simpler baseline and should be the main deployment direction."
    )
    return lines


def add_hyperparameter_page(pdf: PdfPages, finalists_df: pd.DataFrame) -> None:
    lines = []
    for _, row in finalists_df.iterrows():
        hyperparameters = row["Hyperparameters"]
        try:
            parsed = ast.literal_eval(hyperparameters)
            hp_items = [f"{key}={value}" for key, value in parsed.items()]
        except Exception:
            hp_items = [hyperparameters]
        lines.append(f"{row['Model']}")
        lines.append(f"Threshold: {row['Threshold']}")
        lines.append(f"Reason: {row['Selection Reason']}")
        lines.append("Hyperparameters:")
        for item in hp_items:
            lines.append(f"- {item}")
        lines.append("")
    add_text_page(pdf, "Frozen Finalist Settings", lines)


def generate_report(output_path: Path) -> Path:
    dedup_df = load_csv(PREPROCESSING_DIR / "deduplicated_dataset_summary.csv")
    class_df = load_csv(PREPROCESSING_DIR / "class_distribution.csv")
    time_amount_df = load_csv(PREPROCESSING_DIR / "time_amount_summary.csv")
    fraud_time_amount_df = load_csv(PREPROCESSING_DIR / "fraud_time_amount_summary.csv")
    top_bins_df = load_csv(PREPROCESSING_DIR / "top_10_fraud_rate_bins.csv")

    baseline_df = load_csv(TRAIN_DIR / "baseline_model_comparison.csv")
    logreg_threshold_df = load_csv(TRAIN_DIR / "logreg_threshold_results.csv")
    weighted_compare_df = load_csv(TRAIN_DIR / "logreg_vs_weighted_logreg_comparison.csv")

    finalists_df = load_csv(EVALUATE_DIR / "finalists_for_evaluation.csv")
    final_results_df = load_csv(EVALUATE_DIR / "final_test_results.csv")
    final_confusions_df = load_csv(EVALUATE_DIR / "final_confusion_matrices.csv")
    error_summary_df = load_csv(EVALUATE_DIR / "best_model_error_summary.csv")
    error_subset_df = load_csv(EVALUATE_DIR / "best_model_error_subset_summary.csv")
    feature_importance_df = load_csv(EVALUATE_DIR / "best_model_top_feature_importance.csv")
    test_predictions_df = load_csv(EVALUATE_DIR / "test_set_model_predictions.csv")

    preprocessing_notes = read_text(PREPROCESSING_DIR / "preprocessing_notes.txt")
    train_notes = read_text(TRAIN_DIR / "train_notes.txt")
    evaluation_notes = read_text(EVALUATE_DIR / "evaluation_notes.txt")

    if error_summary_df is None:
        error_summary_df = derive_best_model_error_summary(final_results_df, test_predictions_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_path) as pdf:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        best_model_row = final_results_df.iloc[0] if final_results_df is not None and not final_results_df.empty else None

        add_cover_page(pdf, best_model_row, generated_at)
        add_text_page(pdf, "Project Snapshot", build_executive_summary(dedup_df, baseline_df, final_results_df))

        add_section_divider(pdf, "01", "Data and Risk Landscape", "What the dataset looks like and why fraud detection is a precision-recall problem.")

        if dedup_df is not None:
            overview_df = dedup_df.copy()
            overview_df["fraud_percentage"] = overview_df["fraud_proportion"] * 100
            add_bullets_and_table_page(
                pdf,
                "Dataset Overview",
                build_preprocessing_highlights(dedup_df, time_amount_df, fraud_time_amount_df, top_bins_df),
                overview_df[["rows_after_deduplication", "non_fraud_count", "fraud_count", "fraud_percentage"]].round(4),
                table_title="Dataset summary after deduplication",
                subtitle="The data is heavily imbalanced, so model ranking quality matters more than simple accuracy.",
                max_rows=5,
            )

        if class_df is not None:
            add_bullets_and_chart_page(
                pdf,
                "Class Imbalance",
                [
                    "Fraud cases make up only a tiny fraction of the data.",
                    "This imbalance is why PR-AUC became the primary metric throughout the project.",
                    "High ROC-AUC alone is not enough when the positive class is rare.",
                ],
                class_df.assign(Class=class_df["Class"].astype(str), percentage=class_df["proportion"] * 100),
                index_col="Class",
                metric_cols=["percentage"],
                subtitle="Percent share of each class after deduplication.",
            )

        add_two_images_page(
            pdf,
            "Exploratory Visuals: Distribution and Risk Over Time",
            [
                (PREPROCESSING_DIR / "fraud_vs_normal_amount_time.png", "Fraud vs Normal by Amount and Time"),
                (PREPROCESSING_DIR / "fraud_rate_over_time.png", "Fraud Rate Over Time"),
            ],
            subtitle="The visuals show both the scale imbalance and the uneven fraud behavior across the timeline.",
        )

        add_two_images_page(
            pdf,
            "Exploratory Visuals: Transaction Scale",
            [
                (PREPROCESSING_DIR / "amount_vs_time.png", "Amount vs Time"),
                (PREPROCESSING_DIR / "fraud_frequency_over_time.png", "Fraud Frequency Over Time"),
            ],
            subtitle="Amount and event frequency vary substantially across the dataset and can shape how models separate fraud from non-fraud.",
        )

        if time_amount_df is not None and fraud_time_amount_df is not None:
            merged_summary = pd.concat(
                [
                    time_amount_df.rename(columns={"Metric": "Metric"}).assign(Group="Overall"),
                    fraud_time_amount_df.rename(columns={"Metric": "Metric"}).assign(Group="Fraud Only"),
                ],
                ignore_index=True,
            )
            add_table_page(
                pdf,
                "Supporting Summary: Time and Amount",
                merged_summary.round(4),
                subtitle="Reference statistics for the overall dataset and the fraud-only subset.",
                max_rows=6,
                font_size=9,
            )

        add_section_divider(pdf, "02", "Model Development Journey", "How the baseline, threshold tuning, and advanced-model shortlist were established.")

        if baseline_df is not None:
            baseline_df = baseline_df.rename(columns={baseline_df.columns[0]: "Model"})
            add_bullets_and_chart_page(
                pdf,
                "Baseline Models",
                build_modeling_highlights(baseline_df, logreg_threshold_df, finalists_df),
                baseline_df[["Model", "PR-AUC", "F1 Score", "ROC-AUC"]],
                index_col="Model",
                metric_cols=["PR-AUC", "F1 Score", "ROC-AUC"],
                subtitle="Baseline performance from the training-stage notebook.",
            )

        if logreg_threshold_df is not None:
            add_threshold_tradeoff_page(pdf, "Logistic Regression Threshold Tradeoff", logreg_threshold_df)

        if weighted_compare_df is not None:
            add_weighted_comparison_page(pdf, "Weighted vs Unweighted Logistic Regression", weighted_compare_df)

        if finalists_df is not None:
            add_bullets_and_table_page(
                pdf,
                "Finalists Selected for Final Evaluation",
                [
                    "Advanced experimentation narrowed the search to a small set of serious candidates rather than carrying every model forward.",
                    "The final shortlist mixes one simple baseline reference with the strongest tuned ensemble candidates.",
                    "Thresholds were frozen from validation before the test-set evaluation step.",
                ],
                finalists_df[["Model", "Threshold", "Selection Reason"]],
                table_title="Frozen finalists",
                subtitle="These are the exact candidates that were carried into the untouched test-set evaluation.",
                max_rows=8,
            )
            add_hyperparameter_page(pdf, finalists_df)

        add_section_divider(pdf, "03", "Final Evaluation and Decision", "The finalists were refit and judged once on the untouched test set.")

        if final_results_df is not None:
            add_bullets_and_table_page(
                pdf,
                "Final Test Decision",
                build_final_evaluation_highlights(final_results_df, error_summary_df),
                final_results_df[
                    ["Model", "Threshold", "Precision", "Recall", "F1 Score", "PR-AUC", "ROC-AUC"]
                ].round(4),
                table_title="Final test-set ranking",
                subtitle="This is the single most important page in the report.",
                max_rows=8,
            )

            add_bullets_and_chart_page(
                pdf,
                "Final Model Ranking",
                [
                    "Tuned XGBoost leads the final ranking on PR-AUC and also has the strongest overall threshold-based result.",
                    "Tuned LightGBM is the closest backup and confirms that boosting is the strongest model family here.",
                    "The tuned Random Forest remains strong but does not quite match the top boosted models on the test set.",
                ],
                final_results_df[["Model", "PR-AUC", "F1 Score", "Precision", "Recall"]],
                index_col="Model",
                metric_cols=["PR-AUC", "F1 Score", "Precision", "Recall"],
                subtitle="Final test-set comparison across all frozen finalists.",
            )

        if final_confusions_df is not None:
            add_confusion_matrix_page(pdf, final_confusions_df)

        if error_summary_df is not None:
            add_bullets_and_table_page(
                pdf,
                "Winning Model Error Summary",
                [
                    "The winner keeps both false positives and false negatives low on the test set.",
                    "This page gives a direct count-based view of the remaining mistakes.",
                    "It complements PR-AUC and F1 by showing what the chosen threshold actually means operationally.",
                ],
                error_summary_df.round(4),
                table_title="Outcome counts for the winning model",
                subtitle="Derived from the saved test predictions for the best final model.",
                max_rows=8,
            )

        if feature_importance_df is not None:
            add_feature_importance_page(pdf, feature_importance_df, "Winning Model Feature Importance")

        appendix_intro = []
        if preprocessing_notes:
            appendix_intro.extend([line for line in preprocessing_notes.splitlines() if line.strip()])
        if train_notes:
            appendix_intro.extend([""] + [line for line in train_notes.splitlines() if line.strip()])
        if evaluation_notes:
            appendix_intro.extend([""] + [line for line in evaluation_notes.splitlines() if line.strip()])
        if appendix_intro:
            add_section_divider(pdf, "A", "Appendix", "Reference notes and supporting artifacts from the notebooks.")
            add_text_page(pdf, "Notebook Notes", appendix_intro)

        if (ADVANCED_DIR / "rf_search_results.csv").exists():
            rf_search_df = load_csv(ADVANCED_DIR / "rf_search_results.csv")
            if rf_search_df is not None:
                top_rf_search = rf_search_df.sort_values("mean_test_score", ascending=False).head(8)
                add_table_page(
                    pdf,
                    "Appendix: Random Forest Search Snapshot",
                    top_rf_search[["mean_test_score", "std_test_score", "params"]].round(4),
                    subtitle="Top-ranked Random Forest search results retained for reference.",
                    max_rows=8,
                    font_size=8,
                )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report from saved project outputs.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "reports" / "credit_card_fraud_detection_report.pdf",
        help="Path to the PDF report to create.",
    )
    args = parser.parse_args()

    created_path = generate_report(args.output)
    print(f"Created report: {created_path}")


if __name__ == "__main__":
    main()
