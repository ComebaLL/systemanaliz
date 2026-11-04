import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ====================================================================
# 1. ДАННЫЕ
# ====================================================================

# Платежная матрица варианта 6: строки — стратегии A1..A4, столбцы — состояния среды S1..S4
VARIANT_6 = [
    [55, 30, 5, -15],
    [25, 65, 30, 10],
    [15, 35, 70, 25],
    [5, 15, 40, 85],
]

# Допуск при сравнении вещественных чисел: два значения считаем равными,
# если они отличаются меньше чем на EPS (иначе из-за округления ничьи теряются)
EPS = 1e-9

# Значения alpha для таблицы Гурвица: 0.0, 0.1, ..., 1.0 (11 точек).
# np.round убирает хвосты вроде 0.30000000000000004
ALPHAS = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 1)


# ====================================================================
# 2. РАСЧЁТ КРИТЕРИЕВ
# ====================================================================

def argbest(values, maximize=True):
    """
    Возвращает индексы ВСЕХ лучших стратегий.
    Лучшая — с максимальным (maximize=True) или минимальным значением.
    Возвращаем список, а не один индекс, потому что возможна ничья
    (например, в Гурвице при alpha = 0.6 равны A3 и A4).
    """
    v = np.asarray(values, dtype=float)
    target = v.max() if maximize else v.min()
    return [i for i, x in enumerate(v) if abs(x - target) < EPS]


def wald(X):
    """
    Критерий Вальда (максиминный): для каждой стратегии берём худший исход
    (минимум по строке), затем выбираем стратегию с наибольшим таким минимумом.
    Это выбор «на случай, если природа сыграет против нас».
    """
    vals = X.min(axis=1)              # axis=1 — по строкам, т.е. минимум по состояниям среды
    return vals, argbest(vals, True)


def maximax(X):
    """
    Критерий Максимакса (оптимистичный): для каждой стратегии берём лучший исход
    (максимум по строке) и выбираем стратегию, где этот максимум наибольший.
    """
    vals = X.max(axis=1)
    return vals, argbest(vals, True)


def risk_matrix(X):
    """
    Матрица рисков (сожалений) Сэвиджа: R[i][j] = max_i X[i][j] - X[i][j].
    Показывает, сколько мы недополучили в состоянии Sj по сравнению
    с лучшей стратегией именно для этого состояния.
    X.max(axis=0) — максимумы по столбцам; вычитание из матрицы идёт построчно (broadcasting).
    """
    return X.max(axis=0) - X


def savage(X):
    """
    Критерий Сэвиджа (минимаксного риска): для каждой стратегии берём наибольшее
    сожаление (максимум по строке матрицы рисков), выбираем стратегию, у которой
    это наибольшее сожаление минимально.
    Возвращаем также саму матрицу рисков — она нужна для отчёта (пункт 7 задания).
    """
    R = risk_matrix(X)
    vals = R.max(axis=1)
    return vals, argbest(vals, False), R     # False: здесь лучше МЕНЬШЕЕ значение


def hurwitz(X, alpha):
    """
    Критерий Гурвица: H = alpha * (худший исход) + (1 - alpha) * (лучший исход).
    Соглашение методички: alpha = 1 даёт критерий Вальда, alpha = 0 — Максимакс.
    Промежуточные alpha — компромисс между пессимизмом и оптимизмом.
    (Если преподаватель использует обратную запись, поменяйте alpha на 1 - alpha.)
    """
    vals = alpha * X.min(axis=1) + (1 - alpha) * X.max(axis=1)
    return vals, argbest(vals, True)


def names(idx):
    """Превращает список индексов [2, 3] в строку 'A3, A4' (нумерация с единицы)."""
    return ", ".join(f"A{i + 1}" for i in idx)


# ====================================================================
# 3. ГРАФИЧЕСКИЙ ИНТЕРФЕЙС
# ====================================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Принятие решений в условиях неопределенности ")
        self.geometry("1100x760")

        self.n, self.m = 4, 4              # n стратегий (строки), m состояний среды (столбцы)
        self.entries = []                  # сетка полей ввода: entries[i][j] — ячейка матрицы
        self.alpha_var = tk.DoubleVar(value=0.5)   # текущее значение ползунка alpha

        self._build()                      # создаём виджеты
        self.calculate()                   # сразу считаем для матрицы варианта 6

    # ---------------------------------------------------------------- построение окна
    def _build(self):
        # --- верхняя панель: матрица + кнопки + ползунок alpha
        top = ttk.LabelFrame(self, text="Платежная матрица X (строки — стратегии, столбцы — состояния среды)")
        top.pack(fill="x", padx=10, pady=8)

        # Сетка ввода матрицы. Нулевая строка и нулевой столбец — подписи S1..S4 и A1..A4
        grid = ttk.Frame(top)
        grid.pack(side="left", padx=10, pady=8)
        ttk.Label(grid, text="").grid(row=0, column=0)
        for j in range(self.m):
            ttk.Label(grid, text=f"S{j + 1}", width=6, anchor="center").grid(row=0, column=j + 1)
        for i in range(self.n):
            ttk.Label(grid, text=f"A{i + 1}").grid(row=i + 1, column=0, padx=4)
            row = []
            for j in range(self.m):
                e = ttk.Entry(grid, width=7, justify="center")
                e.grid(row=i + 1, column=j + 1, padx=2, pady=2)
                e.insert(0, str(VARIANT_6[i][j]))      # заполняем значениями варианта
                row.append(e)
            self.entries.append(row)

        # Кнопки
        ctrl = ttk.Frame(top)
        ctrl.pack(side="left", padx=30)
        ttk.Button(ctrl, text="Рассчитать", command=self.calculate).pack(fill="x", pady=2)

        # Ползунок alpha: при движении вызывает _on_alpha и пересчитывает Гурвица «на лету»
        alpha_box = ttk.Frame(top)
        alpha_box.pack(side="left", padx=20)
        self.alpha_label = ttk.Label(alpha_box, text="alpha = 0.5")
        self.alpha_label.pack(anchor="w")
        ttk.Scale(alpha_box, from_=0, to=1, length=260, variable=self.alpha_var,
                  command=self._on_alpha).pack()
        ttk.Label(alpha_box, text="alpha=1 → Вальд, alpha=0 → Максимакс", foreground="gray").pack(anchor="w")

        # --- вкладки с результатами
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Вкладка 1: текстовый отчёт со всеми расчётами (матрица рисков, выбор по каждому критерию)
        self.txt = tk.Text(self.tabs, font=("Consolas", 11), wrap="none")
        self.tabs.add(self.txt, text="Расчёты и вывод")

        # Вкладка 2: таблица Гурвица (пункт 5 задания — шаг alpha 0.1)
        hf = ttk.Frame(self.tabs)
        self.tabs.add(hf, text="Гурвиц, шаг 0.1")
        cols = ["alpha"] + [f"A{i + 1}" for i in range(self.n)] + ["best"]
        self.tree = ttk.Treeview(hf, columns=cols, show="headings", height=12)
        heads = ["alpha"] + [f"A{i + 1}" for i in range(self.n)] + ["Оптимальная"]
        for c, h in zip(cols, heads):
            self.tree.heading(c, text=h)
            self.tree.column(c, width=110, anchor="center")
        self.tree.pack(fill="both", expand=True)

        # Вкладка 3: холст matplotlib, встроенный в tkinter (пункт 6 задания — графики)
        gf = ttk.Frame(self.tabs)
        self.tabs.add(gf, text="Графики чувствительности")
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=gf)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------------------------------------------------------- обработчики событий
    def reset(self):
        """Возвращает в поля ввода исходную матрицу варианта 6 и пересчитывает."""
        for i in range(self.n):
            for j in range(self.m):
                self.entries[i][j].delete(0, "end")
                self.entries[i][j].insert(0, str(VARIANT_6[i][j]))
        self.calculate()

    def _on_alpha(self, _=None):
        """
        Вызывается при движении ползунка. Округляем alpha до шага 0.1
        (требование задания), обновляем подпись и пересчитываем.
        read=False — матрицу заново из полей не читаем, она не менялась.
        """
        a = round(self.alpha_var.get() * 10) / 10
        self.alpha_label.config(text=f"α = {a:.1f}")
        if hasattr(self, "X"):             # защита: ползунок может сработать до первого расчёта
            self.calculate(read=False)

    def read_matrix(self):
        """
        Считывает матрицу из полей ввода в numpy-массив.
        Запятую заменяем на точку (удобно для русской раскладки).
        При нечисловом вводе показываем ошибку и возвращаем None.
        """
        try:
            return np.array([[float(self.entries[i][j].get().replace(",", "."))
                              for j in range(self.m)] for i in range(self.n)])
        except ValueError:
            messagebox.showerror("Ошибка", "В матрице должны быть только числа.")
            return None

    # ---------------------------------------------------------------- основной расчёт
    def calculate(self, read=True):
        """Считает все четыре критерия и обновляет три вкладки."""
        if read:
            X = self.read_matrix()
            if X is None:
                return
            self.X = X                     # запоминаем, чтобы ползунок не перечитывал поля
        X = self.X
        a = round(self.alpha_var.get() * 10) / 10

        # Каждый критерий возвращает значения по стратегиям и список оптимальных индексов
        w_vals, w_best = wald(X)
        mm_vals, mm_best = maximax(X)
        s_vals, s_best, R = savage(X)
        h_vals, h_best = hurwitz(X, a)

        self._fill_text(X, R, a, w_vals, w_best, mm_vals, mm_best, s_vals, s_best, h_vals, h_best)
        self._fill_table(X)
        self._draw(X, a, w_vals, w_best, mm_vals, mm_best, s_vals, s_best)

    # ---------------------------------------------------------------- вкладка 1: текстовый отчёт
    def _fill_text(self, X, R, a, wv, wb, mv, mb, sv, sb, hv, hb):
        L = []                             # собираем строки отчёта в список, потом склеиваем

        L.append("Платежная матрица X:")
        L.append("      " + "".join(f"{'S' + str(j + 1):>8}" for j in range(self.m)))
        for i in range(self.n):
            L.append(f"  A{i + 1}  " + "".join(f"{X[i, j]:8.0f}" for j in range(self.m)))

        # Вальд: показываем минимум по каждой строке и выбор максимального из них
        L.append("\n1. Критерий Вальда: W(Ai) = min_j x_ij")
        for i in range(self.n):
            L.append(f"  A{i + 1}: min = {wv[i]:.1f}")
        L.append(f"  -> max = {wv[wb[0]]:.1f}, стратегия {names(wb)}")

        # Максимакс: максимум по каждой строке
        L.append("\n2. Критерий Максимакса: M(Ai) = max_j x_ij")
        for i in range(self.n):
            L.append(f"  A{i + 1}: max = {mv[i]:.1f}")
        L.append(f"  -> max = {mv[mb[0]]:.1f}, стратегия {names(mb)}")

        # Сэвидж: промежуточная матрица рисков + максимальный риск по строкам
        L.append("\n3. Критерий Сэвиджа")
        L.append("  Максимумы столбцов: " + ", ".join(f"{x:.0f}" for x in X.max(axis=0)))
        L.append("  Матрица рисков R = max_i x_ij − x_ij:")
        L.append("        " + "".join(f"{'S' + str(j + 1):>8}" for j in range(self.m)) + f"{'max':>8}")
        for i in range(self.n):
            L.append(f"    A{i + 1}  " + "".join(f"{R[i, j]:8.0f}" for j in range(self.m)) + f"{sv[i]:8.0f}")
        L.append(f"  -> min из максимальных рисков = {sv[sb[0]]:.1f}, стратегия {names(sb)}")

        # Гурвиц: подставляем текущее alpha и показываем формулу с числами
        L.append(f"\n4. Критерий Гурвица: H(Ai) = α·min + (1−α)·max, alpha = {a:.1f}")
        for i in range(self.n):
            L.append(f"  A{i + 1}: H = {a:.1f}·{X[i].min():.0f} + {1 - a:.1f}·{X[i].max():.0f} = {hv[i]:.2f}")
        L.append(f"  -> стратегия {names(hb)}")

        # Ищем, при каких alpha меняется оптимальная стратегия. Идём по сетке alpha;
        # пока оптимум тот же, продлеваем текущий отрезок, иначе начинаем новый.
        prev, segs = None, []
        for al in ALPHAS:
            cur = tuple(hurwitz(X, al)[1])
            if cur != prev:
                segs.append([al, al, cur])     # [начало, конец, оптимальные стратегии]
                prev = cur
            else:
                segs[-1][1] = al
        L.append("\n  Оптимум по alpha (шаг 0.1):")
        for lo, hi, cur in segs:
            L.append(f"    α ∈ [{lo:.1f}; {hi:.1f}] -> {names(cur)}")

        # Сводка и «голосование» критериев: сколько раз каждая стратегия признана лучшей
        L.append("\nСводка:")
        L.append(f"  Вальд: {names(wb)} | Максимакс: {names(mb)} | Сэвидж: {names(sb)} | Гурвиц(alpha={a:.1f}): {names(hb)}")
        votes = {}
        for grp in (wb, mb, sb, hb):
            for i in grp:
                votes[i] = votes.get(i, 0) + 1
        top = max(votes.values())
        L.append("  Чаще всего выбирается: " + names([i for i, v in votes.items() if v == top])
                 + f" ({top} из 4 критериев)")

        # Text по умолчанию редактируемый; на время записи включаем, потом блокируем
        self.txt.config(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "\n".join(L))
        self.txt.config(state="disabled")

    # ---------------------------------------------------------------- вкладка 2: таблица Гурвица
    def _fill_table(self, X):
        """Одна строка таблицы на каждое alpha из сетки: значения H для всех стратегий и оптимум."""
        self.tree.delete(*self.tree.get_children())    # очищаем старые строки
        for al in ALPHAS:
            vals, b = hurwitz(X, al)
            self.tree.insert("", "end",
                             values=[f"{al:.1f}"] + [f"{v:.2f}" for v in vals] + [names(b)])

    # ---------------------------------------------------------------- вкладка 3: графики
    def _draw(self, X, a, wv, wb, mv, mb, sv, sb):
        """
        Четыре графика в сетке 2x2. Для Вальда, Максимакса и Сэвиджа параметра нет,
        поэтому чувствительность к стратегии показываем столбцами (оптимальная — оранжевая).
        Для Гурвица рисуем линии H(alpha): точка пересечения лучших линий — граница смены оптимума.
        """
        self.fig.clear()
        labels = [f"A{i + 1}" for i in range(self.n)]
        base, hi = "#7aa6d8", "#e07a3f"            # обычный цвет и цвет оптимальной стратегии

        def bars(ax, vals, best_idx, title):
            """Столбчатая диаграмма с подсветкой оптимума и подписями значений."""
            colors = [hi if i in best_idx else base for i in range(self.n)]
            ax.bar(labels, vals, color=colors)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.axhline(0, color="black", lw=0.6)    # линия нуля — важна для отрицательных значений
            ax.grid(axis="y", alpha=0.3)

        bars(self.fig.add_subplot(221), wv, wb, "Вальд: min по строке (больше — лучше)")
        bars(self.fig.add_subplot(222), mv, mb, "Максимакс: max по строке (больше — лучше)")
        bars(self.fig.add_subplot(223), sv, sb, "Сэвидж: макс. риск (меньше — лучше)")

        # Гурвиц: плотная сетка из 101 точки даёт гладкие линии (H линейна по alpha)
        ax = self.fig.add_subplot(224)
        fine = np.linspace(0, 1, 101)
        for i in range(self.n):
            ax.plot(fine, fine * X[i].min() + (1 - fine) * X[i].max(), label=labels[i])
        ax.axvline(a, color="gray", ls="--", lw=1)  # пунктир — текущее alpha с ползунка
        ax.set_xlabel("α")
        ax.set_title("Гурвиц: H(Ai) от α", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        self.fig.tight_layout()
        self.canvas.draw()


# Точка входа: создаём окно и запускаем цикл обработки событий
if __name__ == "__main__":
    App().mainloop()