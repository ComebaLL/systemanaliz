import matplotlib.pyplot as plt
import numpy as np

def calculate_eoq(D, Co, Ch, L, Cp=0):
    Q_opt = np.sqrt((2 * D * Co) / Ch)
    N_opt = D / Q_opt
    T_opt = 365 / N_opt
    daily_demand = D / 365
    rop = daily_demand * L

    holding_cost = (Q_opt / 2) * Ch
    ordering_cost = (D / Q_opt) * Co
    purchase_cost = D * Cp
    total_cost = holding_cost + ordering_cost + purchase_cost

    return {
        "Q_opt": round(Q_opt, 2),
        "N_opt": round(N_opt, 2),
        "T_opt": round(T_opt, 2),
        "ROP": round(rop, 2),
        "Total_Cost": round(total_cost, 2)
    }

def plot_inventory_cycles(Q, T, L, ROP):
    cycles = 3
    total_days = int(T * cycles)
    time = np.arange(0, total_days + 1)
    inventory = []

    for t in time:
        current_cycle_time = t % T
        inv_level = Q - (Q / T) * current_cycle_time
        inventory.append(inv_level)

    plt.figure(figsize=(10, 5))
    plt.plot(time, inventory, label='Уровень запаса', color='blue', lw=2)
    plt.axhline(y=ROP, color='red', linestyle='--', label=f'Точка заказа (ROP) = {ROP}')
    plt.title('Модель Уилсона: Изменение уровня запаса во времени (Вариант 6)')
    plt.xlabel('Время (дни)')
    plt.ylabel('Запас на складе (ед.)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Данные варианта 6
D = 9500
Co = 280
Ch = 4.8
L = 4
Cp = 140

result = calculate_eoq(D, Co, Ch, L, Cp)

print("--- Результаты оптимизации запасов (Вариант 6) ---")
print(f"Оптимальный размер заказа (Q*): {result['Q_opt']} ед.")
print(f"Количество заказов в год: {result['N_opt']}")
print(f"Интервал между заказами: {result['T_opt']} дней")
print(f"Точка заказа (ROP): {result['ROP']} ед.")
print(f"Минимальные суммарные затраты на управление: {result['Total_Cost']} руб.")

plot_inventory_cycles(result['Q_opt'], result['T_opt'], L, result['ROP'])