"""Лаб. 7: многокритериальный анализ и принятие решений (tkinter).

Запуск:  uv run python lab7_tk.py
Парето-оптимальность, расстояние до идеала, метод последовательных уступок.
CSV/Excel: 1-й столбец — названия альтернатив, остальные — числовые критерии;
направление критерия — стрелкой ↑/↓ в заголовке или в левой панели.
"""
from __future__ import annotations

import io
import re
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# ═══════════════════════ Вычислительное ядро ═══════════════════════
MAX = "max"
MIN = "min"
EPS = 1e-9


class MCDAError(ValueError):
    """Некорректные входные данные или параметры метода."""


def _check_directions(columns: Sequence[str], directions: Mapping[str, str]) -> None:
    for col in columns:
        if directions.get(col) not in (MAX, MIN):
            raise MCDAError(f"Для критерия «{col}» не задано направление (max/min).")


def to_maximization(table: pd.DataFrame, directions: Mapping[str, str]) -> pd.DataFrame:
    """Приводит все критерии к максимизации (минимизируемые меняют знак)."""
    _check_directions(table.columns, directions)
    out = table.astype(float).copy()
    for col in out.columns:
        if directions[col] == MIN:
            out[col] = -out[col]
    return out


# ───────────────────────── 1. Парето ─────────────────────────
def pareto_analysis(
    table: pd.DataFrame, directions: Mapping[str, str]
) -> tuple[pd.Series, pd.Series]:
    """Возвращает (маска Парето-эффективности, «кем доминируется»).

    j доминирует i, если j не хуже i по всем критериям и строго лучше хотя бы по одному.
    Дубликаты друг друга не доминируют — обе альтернативы остаются эффективными.
    """
    v = to_maximization(table, directions).to_numpy()
    names = list(table.index)
    mask = np.ones(len(v), dtype=bool)
    dominators: list[str] = []
    for i in range(len(v)):
        dom = np.all(v >= v[i], axis=1) & np.any(v > v[i], axis=1)
        mask[i] = not dom.any()
        dominators.append(", ".join(names[j] for j in np.flatnonzero(dom)))
    return (
        pd.Series(mask, index=table.index, name="Парето"),
        pd.Series(dominators, index=table.index, name="Доминируется"),
    )


# ──────────────── 2. Расстояние до «идеала» ────────────────
def normalize(table: pd.DataFrame, directions: Mapping[str, str]) -> pd.DataFrame:
    """Min–max нормирование: 1 — лучшее значение критерия, 0 — худшее.

    Критерий, одинаковый у всех альтернатив, получает z = 1 (как в шаблоне).
    """
    v = to_maximization(table, directions)
    span = v.max() - v.min()
    z = (v - v.min()) / span.where(span > 0, np.nan)
    return z.fillna(1.0)


def validate_weights(
    weights: Mapping[str, float], columns: Sequence[str], tol: float = 1e-6
) -> pd.Series:
    missing = [c for c in columns if c not in weights]
    if missing:
        raise MCDAError(f"Не заданы веса для критериев: {', '.join(missing)}.")
    w = pd.Series({c: float(weights[c]) for c in columns})
    if (w < 0).any():
        raise MCDAError("Веса не могут быть отрицательными.")
    total = round(float(w.sum()), 6)
    if abs(total - 1.0) > tol:
        raise MCDAError(f"Сумма весов должна быть равна 1, сейчас {total:g}.")
    return w


def distance_to_ideal(
    table: pd.DataFrame, directions: Mapping[str, str], weights: Mapping[str, float]
) -> pd.Series:
    """D(x) = sqrt(sum w_i * (1 - z_i(x))^2) — формула из методички."""
    w = validate_weights(weights, list(table.columns))
    z = normalize(table, directions)
    d = np.sqrt(((1.0 - z) ** 2 * w).sum(axis=1))
    return d.rename("D")


# ───────────────── 3. Метод последовательных уступок ─────────────────
@dataclass
class ConcessionStep:
    criterion: str
    direction: str
    best: float
    delta: float
    threshold: float
    kept: list[str]
    dropped: list[str]


@dataclass
class ConcessionResult:
    winner: str
    steps: list[ConcessionStep]
    eliminated_at: dict[str, int] = field(default_factory=dict)  # имя -> номер шага
    ties: list[str] = field(default_factory=list)


def concession_method(
    table: pd.DataFrame,
    directions: Mapping[str, str],
    order: Sequence[str],
    deltas: Mapping[str, float],
) -> ConcessionResult:
    """Последовательные уступки по цепочке приоритетов order (f1 > f2 > ...).

    Шаг k: находим лучшее значение критерия среди оставшихся и сохраняем
    альтернативы, не уступающие ему больше чем на Δ_k. Для последнего критерия
    уступка не применяется: выбирается лучшая альтернатива.
    """
    if not order:
        raise MCDAError("Цепочка приоритетов пуста.")
    if len(set(order)) != len(order):
        raise MCDAError("Критерий встречается в цепочке приоритетов дважды.")
    unknown = [c for c in order if c not in table.columns]
    if unknown:
        raise MCDAError(f"Неизвестные критерии: {', '.join(unknown)}.")
    _check_directions(order, directions)

    alive = list(table.index)
    steps: list[ConcessionStep] = []
    eliminated: dict[str, int] = {}

    for k, crit in enumerate(order, start=1):
        if len(alive) == 1:
            break
        is_max = directions[crit] == MAX
        col = table.loc[alive, crit].astype(float)
        best = float(col.max() if is_max else col.min())
        if k == len(order):
            delta = 0.0
        else:
            if crit not in deltas:
                raise MCDAError(f"Не задана уступка dekta для критерия «{crit}».")
            delta = float(deltas[crit])
            if delta < 0:
                raise MCDAError(f"Уступка delta для «{crit}» не может быть отрицательной.")
        if is_max:
            threshold = best - delta
            keep_mask = col >= threshold - EPS
        else:
            threshold = best + delta
            keep_mask = col <= threshold + EPS
        kept = list(col.index[keep_mask])
        dropped = [n for n in alive if n not in kept]
        for n in dropped:
            eliminated[n] = k
        steps.append(ConcessionStep(crit, directions[crit], best, delta, threshold, kept, dropped))
        alive = kept

    return ConcessionResult(
        winner=alive[0],
        steps=steps,
        eliminated_at=eliminated,
        ties=alive if len(alive) > 1 else [],
    )


# ═══════════════════════ Загрузка данных ═══════════════════════
MAX_BYTES = 5 * 1024 * 1024


class DataError(ValueError):
    """Файл или таблица не прошли проверку."""


MIN_HINTS = ("время", "задержк", "стоимост", "цен", "размер", "энерго", "ram", "озу", "ложн",
             "fpr", "расход", "потер", "latency", "cost", "price", "size", "time", "error", "loss")


def _split_header(header: str) -> tuple[str, str | None]:
    """«ROC-AUC ↑» -> («ROC-AUC», "max"); «Cost [min]» -> («Cost», "min"); без метки -> None."""
    h = str(header).strip()
    for arrow, direction in (("↑", MAX), ("↓", MIN)):
        if arrow in h:
            return re.sub(r"\s*[\(\[]?\s*" + arrow + r"\s*[\)\]]?", "", h).strip(), direction
    m = re.search(r"\s*\[(max|min)\]$", h, re.I)
    if m:
        return h[: m.start()].strip(), m.group(1).lower()
    return h, None


def guess_direction(name: str) -> str:
    """Подсказка для критериев без стрелки (можно изменить в боковой панели)."""
    low = name.lower()
    return MIN if any(k in low for k in MIN_HINTS) else MAX


def _decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1251"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DataError("Не удалось определить кодировку файла (ожидается UTF-8 или Windows-1251).")


def _read_raw(filename: str, raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_BYTES:
        raise DataError("Файл больше 5 МБ.")
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm", ".xls")):
            return pd.read_excel(io.BytesIO(raw), dtype=str)
        text = _decode(raw)
        first = text.lstrip().splitlines()[0] if text.strip() else ""
        sep = max((";", "\t", ","), key=first.count)
        return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
    except DataError:
        raise
    except Exception as exc:  # pandas/openpyxl бросают разные типы
        raise DataError(f"Не удалось прочитать файл: {exc}") from exc


def validate_table(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет таблицу: 1-й столбец — имена, остальные — числовые критерии."""
    if df.shape[1] < 3:
        raise DataError("Нужен столбец с названиями и минимум два столбца критериев.")
    if df.shape[0] < 2:
        raise DataError("Нужно минимум две альтернативы.")
    names = df.iloc[:, 0].astype(str).str.strip()
    if (names == "").any() or df.iloc[:, 0].isna().any():
        raise DataError("В столбце названий есть пустые ячейки.")
    if names.duplicated().any():
        dup = ", ".join(names[names.duplicated()].unique())
        raise DataError(f"Названия альтернатив должны быть уникальны: {dup}.")
    crit = df.iloc[:, 1:]
    if crit.columns.duplicated().any():
        raise DataError("Названия критериев должны быть уникальны.")
    out = {}
    for col in crit.columns:
        s = crit[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = s.astype(str).str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
        num = pd.to_numeric(s, errors="coerce")
        if num.isna().any():
            bad = names[num.isna()].tolist()
            raise DataError(f"Критерий «{col}»: нечисловые или пустые значения у {', '.join(bad)}.")
        if not np.isfinite(num.to_numpy(dtype=float)).all():
            raise DataError(f"Критерий «{col}»: значения должны быть конечными числами.")
        out[col] = num.astype(float)
    result = pd.DataFrame(out)
    result.index = pd.Index(names, name=str(df.columns[0]))
    return result


def load_table(filename: str, raw: bytes) -> tuple[pd.DataFrame, dict[str, str]]:
    """Читает CSV/Excel. Возвращает (таблица, направления из заголовков ↑/↓)."""
    df = _read_raw(filename, raw)
    df = df.dropna(how="all")
    renamed, directions = [], {}
    for i, col in enumerate(df.columns):
        if i == 0:
            renamed.append(str(col).strip())
            continue
        clean, direction = _split_header(col)
        if clean in renamed:
            raise DataError(f"Повторяющееся название критерия: «{clean}».")
        renamed.append(clean)
        if direction:
            directions[clean] = direction
    df.columns = renamed
    return validate_table(df), directions


def to_csv(table: pd.DataFrame, directions: Mapping[str, str]) -> bytes:
    """Экспорт таблицы в CSV, пригодный для повторной загрузки (стрелки в заголовках)."""
    arrow = {MAX: "↑", MIN: "↓"}
    out = table.copy()
    out.columns = [f"{c} {arrow[directions[c]]}" if c in directions else c for c in out.columns]
    return out.to_csv(sep=",", encoding="utf-8-sig").encode("utf-8-sig")



# ═══════════════════════ Графики (matplotlib, без зависимости от tkinter) ═══════════════════════
ARROW = {MAX: "↑", MIN: "↓"}
C_PARETO, C_DOM, C_IDEAL = "#1a9850", "#9e9e9e", "#d73027"
C_METHOD = {"Мин. расстояние": "#2166ac", "Метод уступок": "#f46d43"}


def ideal_point(table: pd.DataFrame, directions: Mapping[str, str]) -> pd.Series:
    """Идеальная точка в исходных единицах: лучшее значение по каждому критерию."""
    return pd.Series({c: (table[c].max() if directions[c] == MAX else table[c].min())
                      for c in table.columns})


def draw_space(fig: Figure, table, directions, axes_cols, pareto, chosen=None) -> None:
    """2D/3D-пространство альтернатив: Парето, доминируемые, идеал, выбор методов."""
    fig.clear()
    if len(axes_cols) not in (2, 3):
        raise MCDAError("Для графика нужно выбрать 2 или 3 критерия.")
    is3d = len(axes_cols) == 3
    ax = fig.add_subplot(111, projection="3d") if is3d else fig.add_subplot(111)
    ideal = ideal_point(table, directions)

    def xyz(frame):
        return [frame[c].to_numpy(dtype=float) for c in axes_cols]

    def label(frame, ax=ax):
        for name, *pt in zip(frame.index, *xyz(frame)):
            if is3d:
                ax.text(*pt, " " + str(name), fontsize=8)
            else:
                ax.annotate(str(name), pt, textcoords="offset points", xytext=(5, 5), fontsize=8)

    par, dom = table[pareto], table[~pareto]
    if len(par):
        ax.scatter(*xyz(par), c=C_PARETO, s=60, label="Парето-эффективные", depthshade=False) if is3d \
            else ax.scatter(*xyz(par), c=C_PARETO, s=60, label="Парето-эффективные")
        label(par)
    if len(dom):
        ax.scatter(*xyz(dom), c=C_DOM, s=45, label="Доминируемые", depthshade=False) if is3d \
            else ax.scatter(*xyz(dom), c=C_DOM, s=45, label="Доминируемые")
        label(dom)

    if len(table.columns) == 2 and not is3d and len(par) > 1:
        front = par.sort_values(axes_cols[0])
        ax.plot(front[axes_cols[0]], front[axes_cols[1]], ":", color=C_PARETO, label="Граница Парето")

    ideal_df = pd.DataFrame([ideal[list(axes_cols)]], index=["Идеал"])
    kw = dict(marker="D", c=C_IDEAL, s=80, label="Идеальная точка")
    ax.scatter(*xyz(ideal_df), **kw, **({"depthshade": False} if is3d else {}))
    label(ideal_df)

    for method, name in (chosen or {}).items():
        if name in table.index:
            pt = table.loc[[name]]
            ax.scatter(*xyz(pt), s=300, facecolors="none", edgecolors=C_METHOD.get(method, "k"),
                       linewidths=2.5, label=f"Выбор: {method}",
                       **({"depthshade": False} if is3d else {}))

    names = [f"{c} {ARROW[directions[c]]}" for c in axes_cols]
    ax.set_xlabel(names[0]); ax.set_ylabel(names[1])
    if is3d:
        ax.set_zlabel(names[2])
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)


def draw_distances(fig: Figure, dist: pd.Series, pareto: pd.Series) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    d = dist.sort_values()
    colors = [C_PARETO if pareto[n] else C_DOM for n in d.index]
    bars = ax.bar([str(n) for n in d.index], d.values, color=colors)
    ax.bar_label(bars, fmt="%.4f", fontsize=8)
    ax.set_ylabel("D (меньше — лучше)")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Расстояние до идеала (зелёные — Парето-эффективные)", fontsize=10)


# ═══════════════════════ Интерфейс (tkinter) ═══════════════════════
DIR_LABELS = {MAX: "↑ максимизировать", MIN: "↓ минимизировать"}
LABEL_TO_DIR = {v: k for k, v in DIR_LABELS.items()}
NO_AXIS = "— (график 2D)"


def parse_float(text: str) -> float:
    try:
        return float(str(text).strip().replace(",", "."))
    except ValueError as exc:
        raise MCDAError(f"«{text}» — не число.") from exc


def fill_tree(tree: ttk.Treeview, df: pd.DataFrame) -> None:
    """Заполняет Treeview таблицей (индекс становится первым столбцом)."""
    frame = df.reset_index()
    cols = [str(c) for c in frame.columns]
    tree.delete(*tree.get_children())
    tree["columns"] = cols
    tree["show"] = "headings"
    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=max(90, min(220, 9 * len(c))), anchor="center", stretch=True)
    tree.column(cols[0], anchor="w")
    for row in frame.itertuples(index=False):
        tree.insert("", "end", values=[f"{v:g}" if isinstance(v, (float, np.floating)) else v for v in row])


def make_tree(parent, height=7) -> ttk.Treeview:
    box = ttk.Frame(parent)
    box.pack(fill="x", padx=4, pady=4)
    tree = ttk.Treeview(box, height=height)
    sb = ttk.Scrollbar(box, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=sb.set)
    tree.pack(fill="x")
    sb.pack(fill="x")
    return tree


def make_canvas(parent) -> tuple[Figure, FigureCanvasTkAgg]:
    """Холст matplotlib с панелью (масштаб, перемещение, вращение 3D мышью, сохранение)."""
    fig = Figure(figsize=(6, 4), dpi=100, layout="constrained")
    holder = ttk.Frame(parent)
    holder.pack(fill="both", expand=True, padx=4, pady=4)
    canvas = FigureCanvasTkAgg(fig, master=holder)
    bar = ttk.Frame(holder)
    bar.pack(side="bottom", fill="x")
    NavigationToolbar2Tk(canvas, bar)
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    return fig, canvas


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Многокритериальный анализ и принятие решений")
        self.geometry("1400x860")
        self.minsize(1100, 700)
        self.table: pd.DataFrame | None = None
        self.file_dirs: dict[str, str] = {}
        self.dir_vars: dict[str, tk.StringVar] = {}
        self.weight_vars: dict[str, tk.DoubleVar] = {}
        self.weight_labels: dict[str, ttk.Label] = {}
        self.order: list[str] = []
        self.in_chain: dict[str, tk.BooleanVar] = {}
        self.delta_vars: dict[str, tk.StringVar] = {}
        self.axis_vars = [tk.StringVar() for _ in range(3)]
        self._building = False
        self._after_id = None
        self.pareto = self.dominators = self.z = self.dist = self.conc = None
        self._build_ui()

    # ───────────────────────── построение окна ─────────────────────────
    def _build_ui(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Button(top, text="Загрузить CSV / Excel", command=self.open_file).pack(side="left")
        self.file_lbl = ttk.Label(top, text="Файл не загружен")
        self.file_lbl.pack(side="left", padx=10)

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(paned, width=360)
        paned.add(left, weight=0)
        self.crit_box = ttk.LabelFrame(left, text="Критерии: направление и вес ωᵢ")
        self.crit_box.pack(fill="x", padx=4, pady=4)
        self.sum_lbl = ttk.Label(left, text="sum_omega = —", font=("TkDefaultFont", 10, "bold"))
        self.sum_lbl.pack(anchor="w", padx=8)
        ttk.Button(left, text="Нормировать веса до sum_omega = 1", command=self.normalize_weights).pack(
            anchor="w", padx=8, pady=4)
        self.axes_box = ttk.LabelFrame(left, text="Оси графика")
        self.axes_box.pack(fill="x", padx=4, pady=8)
        self.msg_lbl = ttk.Label(left, text="", foreground="#b00020", wraplength=340, justify="left")
        self.msg_lbl.pack(anchor="w", padx=8, pady=4)

        self.nb = ttk.Notebook(paned)
        paned.add(self.nb, weight=1)
        tabs = {n: ttk.Frame(self.nb) for n in ("Данные", "Парето", "Расстояние до идеала", "Метод уступок", "Сводка")}
        for n, f in tabs.items():
            self.nb.add(f, text=n)

        # Данные
        ttk.Label(tabs["Данные"], text="Исходная матрица").pack(anchor="w", padx=6, pady=(6, 0))
        self.tree_raw = make_tree(tabs["Данные"], 8)
        ttk.Label(tabs["Данные"], text="Нормированная матрица z (1 — лучшее значение, 0 — худшее)").pack(
            anchor="w", padx=6)
        self.tree_z = make_tree(tabs["Данные"], 8)

        # Парето
        self.tree_pareto = make_tree(tabs["Парето"], 6)
        self.pareto_lbl = ttk.Label(tabs["Парето"], text="", font=("TkDefaultFont", 10, "bold"))
        self.pareto_lbl.pack(anchor="w", padx=6)
        self.fig_p, self.cv_p = make_canvas(tabs["Парето"])

        # Расстояние до идеала
        self.tree_dist = make_tree(tabs["Расстояние до идеала"], 6)
        self.dist_lbl = ttk.Label(tabs["Расстояние до идеала"], text="", font=("TkDefaultFont", 10, "bold"))
        self.dist_lbl.pack(anchor="w", padx=6)
        self.fig_d, self.cv_d = make_canvas(tabs["Расстояние до идеала"])

        # Метод уступок
        t = tabs["Метод уступок"]
        ttk.Label(t, text="Цепочка приоритетов f1 > f2 > …"
                          "(для последнего критерия delta не нужна):", wraplength=900).pack(anchor="w", padx=6, pady=6)
        self.chain_frame = ttk.Frame(t)
        self.chain_frame.pack(fill="x", padx=6)
        self.conc_text = tk.Text(t, height=18, wrap="word", state="disabled")
        self.conc_text.tag_configure("b", font=("TkDefaultFont", 10, "bold"))
        self.conc_text.tag_configure("win", font=("TkDefaultFont", 11, "bold"), foreground="#1a7f37")
        self.conc_text.pack(fill="both", expand=True, padx=6, pady=6)

        # Сводка
        s = tabs["Сводка"]
        self.tree_sum = make_tree(s, 6)
        row = ttk.Frame(s)
        row.pack(fill="x", padx=6)
        self.sum_verdict = ttk.Label(row, text="", font=("TkDefaultFont", 10, "bold"))
        self.sum_verdict.pack(side="left")
        ttk.Button(row, text="Сохранить сводку в CSV", command=self.export_summary).pack(side="right")
        self.fig_s, self.cv_s = make_canvas(s)

    # ───────────────────────── загрузка данных ─────────────────────────
    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Таблица альтернатив",
            filetypes=[("Таблицы", "*.csv *.tsv *.txt *.xlsx *.xls"), ("Все файлы", "*.*")])
        if not path:
            return
        try:
            table, dirs = load_table(Path(path).name, Path(path).read_bytes())
        except (DataError, OSError) as exc:
            messagebox.showerror("Ошибка файла", str(exc))
            return
        self.file_lbl.config(text=f"{Path(path).name} — {len(table)} альтернатив, {table.shape[1]} критериев")
        self.set_table(table, dirs)

    def set_table(self, table: pd.DataFrame, file_dirs: dict[str, str]) -> None:
        self._building = True
        self.table, self.file_dirs = table, file_dirs
        cols = list(table.columns)

        for w in self.crit_box.winfo_children():
            w.destroy()
        self.dir_vars, self.weight_vars, self.weight_labels = {}, {}, {}
        base = round(1 / len(cols), 2)
        for i, c in enumerate(cols):
            ttk.Label(self.crit_box, text=c, wraplength=320, font=("TkDefaultFont", 9, "bold")).grid(
                row=2 * i, column=0, columnspan=3, sticky="w", padx=4, pady=(6, 0))
            dv = tk.StringVar(value=DIR_LABELS[file_dirs.get(c) or guess_direction(c)])
            cb = ttk.Combobox(self.crit_box, textvariable=dv, values=list(DIR_LABELS.values()),
                              state="readonly", width=18)
            cb.grid(row=2 * i + 1, column=0, padx=4, pady=2)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.schedule())
            self.dir_vars[c] = dv
            wv = tk.DoubleVar(value=base if i < len(cols) - 1 else round(1 - base * (len(cols) - 1), 2))
            sc = tk.Scale(self.crit_box, from_=0.0, to=1.0, resolution=0.01, orient="horizontal",
                          variable=wv, length=140, showvalue=False,
                          command=lambda _v, c=c: self._weight_changed(c))
            sc.grid(row=2 * i + 1, column=1, padx=2)
            lbl = ttk.Label(self.crit_box, text=f"{wv.get():.2f}", width=5)
            lbl.grid(row=2 * i + 1, column=2)
            self.weight_vars[c], self.weight_labels[c] = wv, lbl

        for w in self.axes_box.winfo_children():
            w.destroy()
        for i, name in enumerate("XYZ"):
            ttk.Label(self.axes_box, text=name).grid(row=i, column=0, padx=4, pady=2)
            values = cols if i < 2 else [NO_AXIS] + cols
            if i == 2 and len(cols) < 3:
                values = [NO_AXIS]
            default = cols[i] if i < len(cols) and (i < 2 or len(cols) >= 3) else NO_AXIS
            self.axis_vars[i].set(default)
            cb = ttk.Combobox(self.axes_box, textvariable=self.axis_vars[i], values=values,
                              state="readonly", width=34)
            cb.grid(row=i, column=1, padx=4, pady=2)
            cb.bind("<<ComboboxSelected>>", lambda _e: self.schedule())

        self.order = list(cols)
        self.in_chain = {c: tk.BooleanVar(value=True) for c in cols}
        self.delta_vars = {}
        for c in cols:
            span = float(table[c].max() - table[c].min())
            v = tk.StringVar(value=f"{round(0.2 * span, 6):g}")
            v.trace_add("write", lambda *_: self.schedule())
            self.delta_vars[c] = v
        self.rebuild_chain()
        self._building = False
        self.recalc()

    # ───────────────────────── цепочка приоритетов ─────────────────────────
    def rebuild_chain(self) -> None:
        for w in self.chain_frame.winfo_children():
            w.destroy()
        active = [c for c in self.order if self.in_chain[c].get()]
        for i, c in enumerate(self.order):
            ttk.Checkbutton(self.chain_frame, text=f"{i + 1}. {c}", variable=self.in_chain[c],
                            command=self._chain_toggled).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            ttk.Button(self.chain_frame, text="▲", width=3, command=lambda k=i: self.move(k, -1)).grid(row=i, column=1)
            ttk.Button(self.chain_frame, text="▼", width=3, command=lambda k=i: self.move(k, 1)).grid(row=i, column=2)
            ttk.Label(self.chain_frame, text="Δ =").grid(row=i, column=3, padx=(12, 2))
            is_last = bool(active) and c == active[-1]
            state = "disabled" if (is_last or c not in active) else "normal"
            ttk.Entry(self.chain_frame, textvariable=self.delta_vars[c], width=12, state=state).grid(row=i, column=4)
            span = float(self.table[c].max() - self.table[c].min())
            ttk.Label(self.chain_frame, text=f"(разброс по таблице: {span:g})").grid(row=i, column=5, padx=8)

    def move(self, k: int, step: int) -> None:
        j = k + step
        if 0 <= j < len(self.order):
            self.order[k], self.order[j] = self.order[j], self.order[k]
            self.rebuild_chain()
            self.schedule()

    def _chain_toggled(self) -> None:
        self.rebuild_chain()
        self.schedule()

    # ───────────────────────── веса ─────────────────────────
    def _weight_changed(self, c: str) -> None:
        if self._building:
            return
        self.weight_labels[c].config(text=f"{self.weight_vars[c].get():.2f}")
        self._update_sum()
        self.schedule()

    def _weights(self) -> dict[str, float]:
        return {c: round(v.get(), 2) for c, v in self.weight_vars.items()}

    def _update_sum(self) -> bool:
        total = round(sum(self._weights().values()), 6)
        ok = abs(total - 1.0) < 1e-6
        self.sum_lbl.config(text=f"sum_omega = {total:g}" + ("" if ok else "  ≠ 1"),
                            foreground="#1a7f37" if ok else "#b00020")
        return ok

    def normalize_weights(self) -> None:
        if self.table is None:
            return
        w = self._weights()
        total = sum(w.values())
        if total <= 0:
            messagebox.showwarning("Веса", "Сумма весов равна нулю — задайте хотя бы один вес.")
            return
        new = {c: round(x / total, 2) for c, x in w.items()}
        top = max(new, key=new.get)
        new[top] = round(new[top] + 1 - sum(new.values()), 2)
        self._building = True
        for c, x in new.items():
            self.weight_vars[c].set(x)
            self.weight_labels[c].config(text=f"{x:.2f}")
        self._building = False
        self._update_sum()
        self.recalc()

    # ───────────────────────── расчёт ─────────────────────────
    def schedule(self) -> None:
        if self._building or self.table is None:
            return
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(150, self.recalc)

    def directions(self) -> dict[str, str]:
        return {c: LABEL_TO_DIR[v.get()] for c, v in self.dir_vars.items()}

    def axes_cols(self) -> list[str]:
        return [v.get() for v in self.axis_vars if v.get() and v.get() != NO_AXIS]

    def recalc(self) -> None:
        self._after_id = None
        if self.table is None or self._building:
            return
        errors: list[str] = []
        table, dirs = self.table, self.directions()
        weights_ok = self._update_sum()

        self.pareto, self.dominators = pareto_analysis(table, dirs)
        self.z = normalize(table, dirs)

        self.dist = None
        if weights_ok:
            try:
                self.dist = distance_to_ideal(table, dirs, self._weights())
            except MCDAError as exc:
                errors.append(str(exc))
        else:
            errors.append("sum_omega <> 1: расстояние до идеала не посчитано (нажмите «Нормировать»).")

        self.conc = None
        chain = [c for c in self.order if self.in_chain[c].get()]
        try:
            deltas = {c: parse_float(self.delta_vars[c].get()) for c in chain[:-1]}
            self.conc = concession_method(table, dirs, chain, deltas)
        except MCDAError as exc:
            errors.append(f"Метод уступок: {exc}")

        self.msg_lbl.config(text="\n".join(errors))
        self.refresh_views()

    # ───────────────────────── отображение ─────────────────────────
    def chosen(self) -> dict[str, str]:
        out = {}
        if self.dist is not None:
            out["Мин. расстояние"] = self.dist.idxmin()
        if self.conc is not None:
            out["Метод уступок"] = self.conc.winner
        return out

    def refresh_views(self) -> None:
        table, dirs = self.table, self.directions()
        fill_tree(self.tree_raw, table)
        fill_tree(self.tree_z, self.z.round(4))

        view = table.copy()
        view["Парето"] = self.pareto.map({True: "yes", False: ""})
        view["Доминируется"] = self.dominators
        fill_tree(self.tree_pareto, view)
        self.pareto_lbl.config(text="Эффективные альтернативы: " + ", ".join(table.index[self.pareto]))

        for fig, cv in ((self.fig_p, self.cv_p), (self.fig_s, self.cv_s)):
            cols = self.axes_cols()
            try:
                draw_space(fig, table, dirs, cols, self.pareto, self.chosen())
            except MCDAError as exc:
                fig.clear()
                fig.text(0.5, 0.5, str(exc), ha="center")
            cv.draw_idle()

        if self.dist is None:
            self.tree_dist.delete(*self.tree_dist.get_children())
            self.dist_lbl.config(text="Расстояние не посчитано: sum_omega должна быть равна 1.")
            self.fig_d.clear()
            self.cv_d.draw_idle()
        else:
            res = pd.concat([self.z.round(4), self.dist.round(4)], axis=1)
            res["Ранг"] = self.dist.rank(method="min").astype(int)
            fill_tree(self.tree_dist, res.sort_values("D"))
            self.dist_lbl.config(text=f"Минимум D: {self.dist.idxmin()} (D = {self.dist.min():.4f})")
            draw_distances(self.fig_d, self.dist, self.pareto)
            self.cv_d.draw_idle()

        self.render_conc_log()
        self.summary = self.build_summary()
        fill_tree(self.tree_sum, self.summary)
        if self.dist is not None and self.conc is not None:
            same = self.dist.idxmin() == self.conc.winner
            self.sum_verdict.config(
                text=(f"Оба метода выбрали: {self.conc.winner}" if same else
                      f"Методы разошлись: расстояние — {self.dist.idxmin()}, уступки — {self.conc.winner}"))
        else:
            self.sum_verdict.config(text="")

    def render_conc_log(self) -> None:
        t = self.conc_text
        t.config(state="normal")
        t.delete("1.0", "end")
        c = self.conc
        if c is None:
            t.insert("end", "Задайте корректную цепочку приоритетов и уступки.")
        else:
            n = len([x for x in self.order if self.in_chain[x].get()])
            for i, s in enumerate(c.steps, 1):
                sign = "≥" if s.direction == MAX else "≤"
                t.insert("end", f"Шаг {i}. {s.criterion}\n", "b")
                t.insert("end", f"  лучшее значение: {s.best:g}")
                if i < n:
                    t.insert("end", f", delta = {s.delta:g}, порог {sign} {s.threshold:g}")
                t.insert("end", f"\n  остаются: {', '.join(s.kept)}\n")
                if s.dropped:
                    t.insert("end", f"  отсечены: {', '.join(s.dropped)}\n")
                t.insert("end", "\n")
            t.insert("end", f"Выбор методом уступок: {c.winner}\n", "win")
            if c.ties:
                t.insert("end", f"(равнозначны по последнему критерию: {', '.join(c.ties)}; взята первая)\n")
        t.config(state="disabled")

    def build_summary(self) -> pd.DataFrame:
        s = self.table.copy()
        s["Парето"] = self.pareto.map({True: "✓", False: ""})
        if self.dist is not None:
            s["D"] = self.dist.round(4)
            s["Ранг по D"] = self.dist.rank(method="min").astype(int)

        def status(n: str) -> str:
            if self.conc is None:
                return "—"
            if n == self.conc.winner:
                return "выбран"
            return f"отсечён на шаге {self.conc.eliminated_at[n]}" if n in self.conc.eliminated_at else "—"

        s["Метод уступок"] = [status(n) for n in s.index]
        return s.sort_values("D") if self.dist is not None else s

    def export_summary(self) -> None:
        if self.table is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="summary.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.summary.to_csv(path, sep=";", encoding="utf-8-sig")
            except OSError as exc:
                messagebox.showerror("Ошибка сохранения", str(exc))


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()