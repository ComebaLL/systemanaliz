import math
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass
from typing import Any


@dataclass
class Result:
    values: dict[str, Any]
    errors: list[str]


class SteganalysisBlackBox:
    """Учебная статическая модель по статье о CSM и стеганоанализе.

    Источник: Р. А. Солодуха, "Проблема CSM в частотной области изображения:
    новый подход к решению", ИУС, 2026, №1, с. 8–18.
    https://cyberleninka.ru/article/n/problema-csm-v-chastotnoy-oblasti-izobrazheniya-novyy-podhod-k-resheniyu/viewer
    """

    ALGORITHMS = {"nsF5": True, "Другой": False}
    JPEG_QF = {75, 90, 95}

    def process(
        self,
        jpeg_quality: int,              # качество изображения
        embedding_percent: float,       # стегановложение
        distance: str,                  # расстояние
        feature_vector: str,            # вектор признаков
        train_size: int,                # размер выборки
        algorithm: str,                 # стеганоалгоритм
    ) -> Result:
        errors: list[str] = []
        if jpeg_quality not in self.JPEG_QF:
            errors.append("JPEG Quality Factor должен быть 75, 90 или 95.")
        if not 0 <= embedding_percent <= 100:
            errors.append("Размер стегановложения должен быть в диапазоне 0..100 %.")
        if distance not in {"euclidean", "correlation", "braycurtis", "cosine"}:
            errors.append("Недопустимая метрика расстояния.")
        if feature_vector not in {"PEV-274", "CC-PEV-548"}:
            errors.append("Недопустимый вектор признаков.")
        if not 50 <= train_size <= 10000:
            errors.append("Размер выборки должен быть от 50 до 10000.")
        if algorithm not in self.ALGORITHMS:
            errors.append("Неизвестный стеганоалгоритм.")
        if errors:
            return Result({"Статус": "ОШИБКА / РЕЖИМ ЗАЩИТЫ"}, errors)

        # Учебная статическая аппроксимация поведения описанного регрессора.
        # Не является воспроизведением авторского обученного Ridge-моделя.
        qf_bonus = {75: 0.92, 90: 1.00, 95: 1.03}[jpeg_quality]
        distance_factor = {
            "euclidean": 1.00,
            "correlation": 1.02,
            "braycurtis": 1.01,
            "cosine": 1.03,
        }[distance]
        feature_factor = {"PEV-274": 1.02, "CC-PEV-548": 0.99}[feature_vector]
        sample_factor = min(1.0, train_size / 300.0)
        algorithm_factor = 1.0 if algorithm == "nsF5" else 0.92

        predicted = embedding_percent * qf_bonus * distance_factor * feature_factor * algorithm_factor
        predicted = max(0.0, min(100.0, predicted))
        predicted = round(predicted, 2)

        # Выходы учебной модели. R^2 из статьи показываем как справочное
        # экспериментальное значение, а не как результат этой аппроксимации.
        values = {
            "Статус": "НОРМАЛЬНОЕ ИСПОЛНЕНИЕ",
            "Прогноз размера вложения": f"{predicted} %",
            "Оценка размера вложения": f"{predicted} %",
            "R^2 из эксперимента статьи": "= 0.89 (справочно)",
            "Используемая метрика": distance,
            "Размер обучающей выборки": str(train_size),
        }
        return Result(values, [])


class MatrixBlackBox:
    """Учебная статическая модель на основе статьи о бициклических матрицах.
    https://cyberleninka.ru/article/n/vzaimosvyaz-simmetriy-bitsiklicheskih-ortogonalnyh-matrits-i-ih-poryadkov/viewer
    """

    def process(self, family: str, t: int, d: int) -> Result:
        """
        t - размер матрицы
        d - определяет диагональные значения
        """
        errors: list[str] = []
        if family not in {"Один 4t-1", "Один 4t-3"}:
            errors.append("Выберите тип матрицы Одина.")
        if t < 2:
            errors.append("t должен быть не меньше 2.")
        if d not in {0, 1}:
            errors.append("d должен быть 0 или 1 для учебной модели.")
        if errors:
            return Result({"Статус": "ОШИБКА / РЕЖИМ ЗАЩИТЫ"}, errors)

        if family == "Один 4t-1":
            n = 4 * t - 1
            symmetry = "симметричная / кососимметричная форма"
        else:
            n = 4 * t - 3
            symmetry = "симметричная"

        v = (n - 1) // 2
        b = 1 - 2 * d

        # Учебная конструкция квадратной матрицы: циклический шаблон со знаками.
        m = []
        for i in range(n):
            row = []
            for j in range(n):
                k = (j - i) % n
                val = 1 if k <= v else b
                if i == j:
                    val = d
                row.append(val)
            m.append(row)

        # Проверка симметрии/кососимметрии и простой ортогональности по H^T H.
        symmetric = all(m[i][j] == m[j][i] for i in range(n) for j in range(n))
        skew = all(m[i][j] == -m[j][i] for i in range(n) for j in range(n) if i != j)
        gram_diag = []
        gram_off = []
        for i in range(n):
            for j in range(n):
                dot = sum(m[k][i] * m[k][j] for k in range(n))
                if i == j:
                    gram_diag.append(dot)
                else:
                    gram_off.append(abs(dot))
        max_off = max(gram_off) if gram_off else 0

        preview = "\n".join(" ".join(f"{x:2d}" for x in row) for row in m[: min(8, n)])
        if n > 8:
            preview += "\n..."

        values = {
            "Статус": "НОРМАЛЬНОЕ ИСПОЛНЕНИЕ",
            "Порядок матрицы n": str(n),
            "Параметр t": str(t),
            "Параметр v=(n-1)/2": str(v),
            "Параметр b": str(b),
            "Симметрия": symmetry,
            "Фактическая симметрия шаблона": "да" if symmetric else "нет",
            "Макс. внедиагональный элемент HᵀH": str(max_off),
            "Фрагмент матрицы": preview,
        }
        return Result(values, [])


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Лабораторная работа 1 Вар. 6")
        self.geometry("1050x760")
        self.minsize(900, 650)
        self.steg = SteganalysisBlackBox()
        self.matrix = MatrixBlackBox()
        self._build_style()
        self._build_ui()

    def _build_style(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="Лабораторная работа №1 — вариант 6", style="Title.TLabel").pack(anchor="w")
        ttk.Label(root, text="Статические модели «чёрного ящика» для двух объектов исследования").pack(anchor="w", pady=(0, 12))

        self.tabs = ttk.Notebook(root)
        self.tabs.pack(fill="both", expand=True)

        self.steg_tab = ttk.Frame(self.tabs, padding=12)
        self.matrix_tab = ttk.Frame(self.tabs, padding=12)
        self.test_tab = ttk.Frame(self.tabs, padding=12)
        self.tabs.add(self.steg_tab, text="Объект 1: стеганоанализ")
        self.tabs.add(self.matrix_tab, text="Объект 2: матрицы")
        self.tabs.add(self.test_tab, text="Режим тестирования")

        self._build_steg_tab()
        self._build_matrix_tab()
        self._build_test_tab()

    def _entry(self, parent, row, label, default=""):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=5)
        e = ttk.Entry(parent, width=32)
        e.insert(0, default)
        e.grid(row=row, column=1, sticky="ew", pady=5)
        return e

    def _build_steg_tab(self) -> None:
        tab = self.steg_tab
        left = ttk.Frame(tab)
        left.pack(side="left", fill="y", padx=(0, 18))
        right = ttk.Frame(tab)
        right.pack(side="left", fill="both", expand=True)
        left.columnconfigure(1, weight=1)

        ttk.Label(left, text="Входные параметры X", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.qf = self._entry(left, 1, "JPEG QF", "90")
        self.embedding = self._entry(left, 2, "Вложение, %", "30")
        self.train_size = self._entry(left, 3, "Размер выборки", "300")

        ttk.Label(left, text="Стеганоалгоритм").grid(row=4, column=0, sticky="w", pady=5)
        self.alg = ttk.Combobox(left, values=["nsF5", "Другой"], state="readonly", width=29)
        self.alg.set("nsF5")
        self.alg.grid(row=4, column=1, pady=5)

        ttk.Label(left, text="Метрика расстояния").grid(row=5, column=0, sticky="w", pady=5)
        self.dist = ttk.Combobox(left, values=["euclidean", "correlation", "braycurtis", "cosine"], state="readonly", width=29)
        self.dist.set("euclidean")
        self.dist.grid(row=5, column=1, pady=5)

        ttk.Label(left, text="Вектор признаков").grid(row=6, column=0, sticky="w", pady=5)
        self.features = ttk.Combobox(left, values=["PEV-274", "CC-PEV-548"], state="readonly", width=29)
        self.features.set("PEV-274")
        self.features.grid(row=6, column=1, pady=5)

        ttk.Button(left, text="Рассчитать", style="Accent.TButton", command=self.run_steg).grid(row=7, column=0, columnspan=2, sticky="ew", pady=12)
        ttk.Button(left, text="Очистить", command=lambda: self.result_steg.delete("1.0", "end")).grid(row=8, column=0, columnspan=2, sticky="ew")

        ttk.Label(right, text="Выходы Y", style="Section.TLabel").pack(anchor="w")
        self.result_steg = tk.Text(right, wrap="word", font=("Consolas", 10))
        self.result_steg.pack(fill="both", expand=True, pady=8)

    def _build_matrix_tab(self) -> None:
        tab = self.matrix_tab
        top = ttk.Frame(tab)
        top.pack(fill="x")
        ttk.Label(top, text="Входные параметры X", style="Section.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(top, text="Тип матрицы").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        self.family = ttk.Combobox(top, values=["Один 4t-1", "Один 4t-3"], state="readonly", width=28)
        self.family.set("Один 4t-1")
        self.family.grid(row=1, column=1, sticky="w", pady=5)
        ttk.Label(top, text="t").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        self.t_entry = ttk.Entry(top, width=30)
        self.t_entry.insert(0, "3")
        self.t_entry.grid(row=2, column=1, sticky="w", pady=5)
        ttk.Label(top, text="d").grid(row=3, column=0, sticky="w", padx=(0, 10), pady=5)
        self.d_entry = ttk.Entry(top, width=30)
        self.d_entry.insert(0, "0")
        self.d_entry.grid(row=3, column=1, sticky="w", pady=5)
        ttk.Button(top, text="Построить / рассчитать", style="Accent.TButton", command=self.run_matrix).grid(row=4, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(tab, text="Выходы Y", style="Section.TLabel").pack(anchor="w", pady=(10, 0))
        self.result_matrix = tk.Text(tab, wrap="word", font=("Consolas", 10))
        self.result_matrix.pack(fill="both", expand=True, pady=8)

    def _build_test_tab(self) -> None:
        tab = self.test_tab
        ttk.Label(tab, text="Автоматическое Black Box Testing", style="Section.TLabel").pack(anchor="w")
        ttk.Label(tab, text="Проверяются нормальные, граничные и ошибочные входы для обеих моделей.").pack(anchor="w", pady=(0, 8))
        ttk.Button(tab, text="Запустить все тесты", style="Accent.TButton", command=self.run_tests).pack(anchor="w")
        self.test_result = tk.Text(tab, wrap="word", font=("Consolas", 10))
        self.test_result.pack(fill="both", expand=True, pady=10)

    @staticmethod
    def _show_result(widget: tk.Text, result: Result) -> None:
        widget.delete("1.0", "end")
        for key, value in result.values.items():
            widget.insert("end", f"{key}: {value}\n")
        if result.errors:
            widget.insert("end", "\nОшибки:\n")
            for err in result.errors:
                widget.insert("end", f"- {err}\n")

    def run_steg(self) -> None:
        try:
            result = self.steg.process(
                jpeg_quality=int(self.qf.get()),
                embedding_percent=float(self.embedding.get()),
                distance=self.dist.get(),
                feature_vector=self.features.get(),
                train_size=int(self.train_size.get()),
                algorithm=self.alg.get(),
            )
            self._show_result(self.result_steg, result)
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Числовые поля должны содержать корректные числа.")

    def run_matrix(self) -> None:
        try:
            result = self.matrix.process(self.family.get(), int(self.t_entry.get()), int(self.d_entry.get()))
            self._show_result(self.result_matrix, result)
        except ValueError:
            messagebox.showerror("Ошибка ввода", "t и d должны быть целыми числами.")


    """
    тесты
    """
    def run_tests(self) -> None:
        out: list[str] = []
        cases = [
            ("Стеганоанализ: нормальный", lambda: self.steg.process(90, 30, "euclidean", "PEV-274", 300, "nsF5")),
            ("Стеганоанализ: граничный QF", lambda: self.steg.process(75, 0, "euclidean", "PEV-274", 50, "nsF5")),
            ("Стеганоанализ: ошибочный QF", lambda: self.steg.process(80, 30, "euclidean", "PEV-274", 300, "nsF5")),
            ("Стеганоанализ: ошибочный QF", lambda: self.steg.process(80, -5, "euclidean", "PEV-274", 300, "nsF5")),
            ("Матрица: нормальный", lambda: self.matrix.process("Один 4t-1", 3, 0)),
            ("Матрица: другой класс", lambda: self.matrix.process("Один 4t-3", 3, 0)),
            ("Матрица: ошибочный t", lambda: self.matrix.process("Один 4t-1", 1, 0)),
            ("Матрица: ошибочный t", lambda: self.matrix.process("Один 1t-1", 1, 0)),
        ]
        for name, fn in cases:
            res = fn()
            ok = bool(res.values.get("Статус")) and (not res.errors if "ошибоч" not in name.lower() else bool(res.errors))
            out.append(f"[{ 'OK' if ok else 'FAIL' }] {name}")
            if res.errors:
                out.extend(f"    {e}" for e in res.errors)
        self.test_result.delete("1.0", "end")
        self.test_result.insert("end", "\n".join(out))


if __name__ == "__main__":
    App().mainloop()
