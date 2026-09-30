"""
Лабораторная работа № 2. Сложные системы. Принцип обратной связи
Вариант 6: Разделение труда (Специализация подсистем)

Разведчики (30%) — иммунитет к феромонам, активно исследуют пространство.
Рабочие (70%)   — строго следуют феромоновым тропам.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# ==============================================================
# 1. ДЕКОМПОЗИЦИЯ: Класс Агент (Муравей)
# ==============================================================

class Ant:
    """Элементарный агент сложной системы."""

    def __init__(self, nest_x, nest_y, is_scout=False):
        self.x = nest_x
        self.y = nest_y
        self.has_food = False
        self.is_scout = is_scout          # True = Разведчик, False = Рабочий

    def move(self, grid_size, pheromone_grid, food_grid, nest_x, nest_y):
        if not self.has_food:
            # Нашли еду — берём
            if food_grid[self.x, self.y] > 0:
                food_grid[self.x, self.y] -= 1
                self.has_food = True
                return
            self._move_towards_pheromone(grid_size, pheromone_grid, nest_x, nest_y)
        else:
            # Несём еду домой
            if self.x == nest_x and self.y == nest_y:
                self.has_food = False
                return
            dx = np.sign(nest_x - self.x)
            dy = np.sign(nest_y - self.y)
            self.x = int(np.clip(self.x + dx, 0, grid_size - 1))
            self.y = int(np.clip(self.y + dy, 0, grid_size - 1))

    def _move_towards_pheromone(self, grid_size, pheromone_grid, nest_x, nest_y):
        """
        Разведчик → хаос + лёгкий уход от гнезда (исследование)
        Рабочий   → сильное следование феромону
        """
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]
        valid_moves = []
        weights = []

        for dx, dy in moves:
            nx, ny = self.x + dx, self.y + dy
            if 0 <= nx < grid_size and 0 <= ny < grid_size:
                valid_moves.append((nx, ny))

                if self.is_scout:
                    # Разведчик: игнорирует феромон + bias от гнезда
                    dist_old = abs(self.x - nest_x) + abs(self.y - nest_y)
                    dist_new = abs(nx - nest_x) + abs(ny - nest_y)
                    bias = 1.5 if dist_new > dist_old else 0.6
                    weights.append(bias)
                else:
                    # Рабочий: сильно следует тропе
                    weight = 0.01 + (pheromone_grid[nx, ny] ** 1.4)
                    weights.append(max(weight, 1e-6))

        if not valid_moves:
            return

        chosen_idx = random.choices(range(len(valid_moves)), weights=weights)[0]
        self.x, self.y = valid_moves[chosen_idx]


# ==============================================================
# 2. АГРЕГИРОВАНИЕ: Макросистема
# ==============================================================

class AntColonySystem:
    def __init__(self, size=30, num_ants=120,
                 evaporation_rate=0.04, pheromone_deposit=3.5,
                 scout_ratio=0.20, specialized=True):
        self.size = size
        self.evaporation_rate = evaporation_rate      # отрицательная ОС
        self.pheromone_deposit = pheromone_deposit    # положительная ОС
        self.specialized = specialized

        self.nest = (10, 5)
        self.food_sources = [(15, 18), (25, 13)]

        self.pheromone_grid = np.zeros((size, size))
        self.food_grid = np.zeros((size, size))
        self._init_food()

        # Создание популяции
        self.ants = []
        n_scouts = int(num_ants * scout_ratio) if specialized else 0
        for i in range(num_ants):
            is_scout = (i < n_scouts) if specialized else False
            self.ants.append(Ant(self.nest[0], self.nest[1], is_scout=is_scout))

        self.total_food_collected = 0
        self.food_history = []
        self.pheromone_sum_history = []

    def _init_food(self):
        for fx, fy in self.food_sources:
            self.food_grid[max(0, fx-1):fx+2, max(0, fy-1):fy+2] = 200

    def update_system(self):
        # Отрицательная обратная связь — испарение феромона
        self.pheromone_grid *= (1.0 - self.evaporation_rate)

        for ant in self.ants:
            was_carrying = ant.has_food
            ant.move(self.size, self.pheromone_grid, self.food_grid,
                     self.nest[0], self.nest[1])

            # Учёт доставленной еды
            if was_carrying and not ant.has_food:
                self.total_food_collected += 1

            # Положительная обратная связь — отложение феромона
            if ant.has_food:
                self.pheromone_grid[ant.x, ant.y] += self.pheromone_deposit

        self.food_history.append(self.total_food_collected)
        self.pheromone_sum_history.append(float(np.sum(self.pheromone_grid)))


# ==============================================================
# 3. ВИЗУАЛИЗАЦИЯ
# ==============================================================

def make_display_image(system):
    img = np.zeros((system.size, system.size, 3))
    # Красный — феромон
    img[:, :, 0] = np.clip(system.pheromone_grid / 10.0, 0, 1)
    # Зелёный — еда
    img[:, :, 1] = np.clip(system.food_grid / 200.0, 0, 1)
    # Синий — гнездо
    img[system.nest[0], system.nest[1], 2] = 1.0
    

    for ant in system.ants:
        if ant.is_scout:
            img[ant.x, ant.y] = [0.2, 0.85, 1.0]   # голубые — разведчики
        else:
            img[ant.x, ant.y] = [1.0, 0.9, 0.1]    # белые — рабочие
    return img


def run_live_simulation(specialized=True, steps=400, seed=3):
    random.seed(seed)
    np.random.seed(seed)

    system = AntColonySystem(
        size=30,
        num_ants=120,
        evaporation_rate=0.04,
        pheromone_deposit=3.5,
        scout_ratio=0.30,
        specialized=specialized
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 7))

    key_steps = {0, 50, 200, steps - 1}
    snapshots = {}

    mode = "Разведчики 20% + Рабочие 80%" if specialized else "Все муравьи одинаковые (база)"

    for step in range(steps):
        system.update_system()

        img = make_display_image(system)

        if step in key_steps:
            snapshots[step] = img.copy()

        ax.clear()
        ax.imshow(img, origin='lower')
        ax.set_title(
            f"Вариант 6 | {mode}\n"
            f"Шаг: {step}   |   Собрано пищи: {system.total_food_collected}",
            fontsize=12
        )
        ax.axis('off')
        plt.pause(0.01)

    plt.ioff()
    return system, snapshots


# ==============================================================
# 4. ГЛАВНЫЙ БЛОК
# ==============================================================

if __name__ == "__main__":
    STEPS = 400

    # Специализированная система 
    print("\nЗапускаем специализированную систему (Разведчики 20%)...")
    sys_spec, snaps_spec = run_live_simulation(specialized=True, steps=STEPS, seed=3)

    # Базовая система (для сравнения, без анимации) 
    print("Считаем базовую систему для сравнения...")
    random.seed(3)
    np.random.seed(3)
    sys_base = AntColonySystem(size=30, num_ants=120, evaporation_rate=0.04,
                               pheromone_deposit=3.5, scout_ratio=0, specialized=False)
    for _ in range(STEPS):
        sys_base.update_system()

    # График 1: сбор пищи 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sys_spec.food_history, label='Специализация (Разведчики 20%)', linewidth=2.2, color='#1f77b4')
    ax.plot(sys_base.food_history, label='База (все одинаковые)', linewidth=2.2, color='#ff7f0e', linestyle='--')
    ax.set_xlabel('Шаг симуляции', fontsize=12)
    ax.set_ylabel('Собрано пищи (единиц)', fontsize=12)
    ax.set_title('Динамика сбора ресурсов', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('var6_graph1_food_collection.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Сохранён график 1: var6_graph1_food_collection.png")

    # График 2: сила троп
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sys_spec.pheromone_sum_history, label='Специализация (Разведчики 20%)', linewidth=2.2, color='#2ca02c')
    ax.plot(sys_base.pheromone_sum_history, label='База (все одинаковые)', linewidth=2.2, color='#d62728', linestyle='--')
    ax.set_xlabel('Шаг симуляции', fontsize=12)
    ax.set_ylabel('Суммарная интенсивность феромона', fontsize=12)
    ax.set_title('Формирование и поддержание троп', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('var6_graph2_pheromone_strength.png', dpi=140, bbox_inches='tight')
    plt.close()
    print("Сохранён график 2: var6_graph2_pheromone_strength.png")

    # Итоги 
    print("\n" + "=" * 60)
    print("ИТОГИ")
    print("=" * 60)
    print(f"Специализированная система: {sys_spec.total_food_collected} ед. пищи")
    print(f"Базовая система:            {sys_base.total_food_collected} ед. пищи")
    gain = sys_spec.total_food_collected - sys_base.total_food_collected
    if sys_base.total_food_collected > 0:
        pct = 100.0 * gain / sys_base.total_food_collected
        print(f"Прирост эффективности:      {gain:+d} ед. ({pct:+.1f}%)")
    plt.show()