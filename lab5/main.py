import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from scipy.optimize import linprog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

N, M = 3, 3  # изделий, материалов (сетка 3x3 из варианта)

# Вариант 6
DEFAULT_C = [3, 6, 7]
DEFAULT_A = [[8, 10, 20],
             [4, 13, 8],
             [2, 18, 12]]
DEFAULT_B = [800, 520, 940]

EPS = 1e-6


def solve_lp(c, A, b):
    """max c*x  при A x <= b, x >= 0. Возвращает словарь с результатом."""
    c, A, b = np.array(c, float), np.array(A, float), np.array(b, float)
    res = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, None)] * len(c), method="highs")
    out = {"status": res.status, "message": res.message}
    if res.status == 0:
        x = res.x
        used = A @ x
        out.update(x=x, profit=float(c @ x), used=used, left=b - used,
                   duals=-res.ineqlin.marginals)  # теневые цены
    return out


STATUS_TEXT = {
    0: "Решение успешно найдено",
    2: "Задача не имеет допустимых решений",
    3: "Целевая функция не ограничена",
}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Лаб. работа №5. Оптимизация производственного плана")
        self.minsize(1100, 720)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # запрет отрицательных чисел и букв прямо при вводе
        self.vcmd = (self.register(self._validate_key), "%P")

        left = ttk.Frame(self, padding=10)
        left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(self, padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        self._build_inputs(left)
        self._build_results(right)
        self.reset()

    # ввод 
    @staticmethod
    def _validate_key(new):
        if new == "":
            return True
        if new.count(".") > 1 or new.startswith("-"):
            return False
        return new.replace(".", "", 1).isdigit()

    def _entry(self, parent, r, c):
        e = ttk.Entry(parent, width=8, justify="center",
                      validate="key", validatecommand=self.vcmd)
        e.grid(row=r, column=c, padx=3, pady=3)
        return e

    def _build_inputs(self, f):
        ttk.Label(f, text="Исходные данные", font=("", 12, "bold")).grid(
            row=0, column=0, columnspan=6, sticky="w", pady=(0, 6))

        ttk.Label(f, text="Цена изделия").grid(row=1, column=0, sticky="w")
        self.c_e = [self._entry(f, 1, 1 + j) for j in range(N)]
        for j in range(N):
            ttk.Label(f, text=f"P{j+1}").grid(row=2, column=1 + j)

        ttk.Label(f, text="Нормы расхода, кг", font=("", 10, "bold")).grid(
            row=3, column=0, columnspan=6, sticky="w", pady=(12, 2))
        self.a_e, self.b_e = [], []
        for i in range(M):
            ttk.Label(f, text=f"S{i+1}").grid(row=4 + i, column=0, sticky="w")
            self.a_e.append([self._entry(f, 4 + i, 1 + j) for j in range(N)])
            self.b_e.append(self._entry(f, 4 + i, 1 + N + 1))
        ttk.Label(f, text="Запас, кг").grid(row=3, column=N + 2)

        btns = ttk.Frame(f)
        btns.grid(row=8, column=0, columnspan=6, pady=14, sticky="we")
        ttk.Button(btns, text="Рассчитать", command=self.calculate).pack(
            side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="Очистить", command=self.reset).pack(
            side="left", expand=True, fill="x", padx=2)

        self.status = tk.StringVar(value="Статус: ожидание расчёта")
        ttk.Label(f, textvariable=self.status, wraplength=300,
                  font=("", 10, "bold")).grid(row=9, column=0, columnspan=6, sticky="w")

    def _set(self, entry, val):
        entry.delete(0, tk.END)
        entry.insert(0, str(val))

    def reset(self):
        for j in range(N):
            self._set(self.c_e[j], DEFAULT_C[j])
        for i in range(M):
            self._set(self.b_e[i], DEFAULT_B[i])
            for j in range(N):
                self._set(self.a_e[i][j], DEFAULT_A[i][j])
        self.status.set("Статус: параметры сброшены к варианту 6")
        if hasattr(self, "tree"):
            self._clear_results()

    def _read(self):
        def num(e, name):
            s = e.get().strip()
            if s == "":
                raise ValueError(f"Пустое поле: {name}")
            try:
                v = float(s)
            except ValueError:
                raise ValueError(f"Нужно число в поле: {name}")
            if v < 0:
                raise ValueError(f"Отрицательное значение в поле: {name}")
            return v

        c = [num(self.c_e[j], f"цена P{j+1}") for j in range(N)]
        b = [num(self.b_e[i], f"запас S{i+1}") for i in range(M)]
        A = [[num(self.a_e[i][j], f"норма a{i+1}{j+1}") for j in range(N)] for i in range(M)]
        return c, A, b

    # вывод
    def _build_results(self, f):
        box = ttk.LabelFrame(f, text="Результаты", padding=6)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(0, weight=1)
        cols = ("name", "v1", "v2", "v3", "v4")
        self.tree = ttk.Treeview(box, columns=cols, show="headings", height=14)
        heads = ["Показатель", "Значение / Израсходовано", "Запас", "Остаток / прибыли",
                 "Статус / теневая цена"]
        for c, h, w in zip(cols, heads, (200, 190, 90, 150, 200)):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=w, anchor="center" if c != "name" else "w")
        self.tree.grid(row=0, column=0, sticky="ew")

        self.fig = Figure(figsize=(7, 3.6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=f)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", pady=(8, 0))

    def _clear_results(self):
        self.tree.delete(*self.tree.get_children())
        self.ax.clear()
        self.canvas.draw()

    def _row(self, *vals, tag=""):
        self.tree.insert("", "end", values=vals, tags=(tag,))

    def calculate(self):
        try:
            c, A, b = self._read()
        except ValueError as e:
            messagebox.showwarning("Ошибка ввода", str(e))
            return
        try:
            r = solve_lp(c, A, b)
        except Exception as e:  
            messagebox.showerror("Ошибка расчёта", str(e))
            return

        self._clear_results()
        self.status.set("Статус: " + STATUS_TEXT.get(r["status"], r["message"]))
        if r["status"] != 0:
            return

        self.tree.tag_configure("head", background="#e8eefc")
        self._row("ОПТИМАЛЬНЫЙ ПЛАН", "", "", "", "", tag="head")
        for j, v in enumerate(r["x"]):
            self._row(f"Изделие P{j+1}", f"{v:.2f} ед.", "", "", "")
        self._row("Максимальная стоимость выпуска", f"{r['profit']:.2f} у.е.", "", "", "", tag="head")

        self._row("ИСПОЛЬЗОВАНИЕ МАТЕРИАЛОВ", "", "", "", "", tag="head")
        for i in range(M):
            deficit = r["left"][i] < EPS
            self._row(f"Материал S{i+1}", f"{r['used'][i]:.2f}", f"{b[i]:.2f}",
                      f"{max(r['left'][i], 0):.2f}",
                      ("ДЕФИЦИТ" if deficit else "ИЗБЫТОК") + f" (y={r['duals'][i]:.4f})")

        self._row("Анализ чувствительности (delta F при изменении запаса)", "", "", "", "", tag="head")
        for i in range(M):
            cells = []
            for k in (-0.1, 0.1):
                b2 = list(b)
                b2[i] = b[i] * (1 + k)
                r2 = solve_lp(c, A, b2)
                cells.append(f"{r2['profit'] - r['profit']:+.2f}" if r2["status"] == 0 else "—")
            self._row(f"Запас S{i+1}", "−10%: " + cells[0], "", "+10%: " + cells[1], "")

        self._plot(b, r["used"])

    def _plot(self, b, used):
        self.ax.clear()
        x = np.arange(M)
        w = 0.38
        self.ax.bar(x - w / 2, b, w, label="Доступный запас", color="#3d5afe")
        self.ax.bar(x + w / 2, used, w, label="Фактический расход", color="#e07b00")
        for xi, (bi, ui) in enumerate(zip(b, used)):
            self.ax.text(xi - w / 2, bi, f"{bi:.0f}", ha="center", va="bottom", fontsize=9)
            self.ax.text(xi + w / 2, ui, f"{ui:.0f}", ha="center", va="bottom", fontsize=9)
        self.ax.set_xticks(x)
        self.ax.set_xticklabels([f"Материал S{i+1}" for i in range(M)])
        self.ax.set_ylabel("Количество, кг")
        self.ax.set_title("Анализ использования материалов ")
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    App().mainloop()