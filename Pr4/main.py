import pandas as pd
import numpy as np
from scipy.optimize import minimize
import time

# --- Параметри ---
days = 180 # Кількість днів для генерації даних
n_hours = 24  # Кількість годин у добі
total_hours = days * n_hours # Загальна кількість годин у періоді

base_load = 200  # Базове навантаження
load_std = 10 # Стандартне відхилення навантаження
price_peak = 0.20 # Ціна в пікові години
price_offpeak = 0.10 # Ціна в не пікові години

# --- Генерація даних на місяць ---

# Генеруємо часові мітки на весь період
timestamp = pd.date_range(start="2024-01-01", periods=total_hours, freq="H")

# Генеруємо добовий патерн навантаження (синусоїда)
daily_load_pattern = base_load + np.sin(np.linspace(0, 2 * np.pi, n_hours)) * base_load * 0.5
# Повторюємо добовий патерн на весь місяць і додаємо шум
load_forecast = np.tile(daily_load_pattern, days) + np.random.normal(0, load_std, total_hours)
# Переконуємось, що навантаження не від'ємне (хоча з base_load=100 це малоймовірно)
load_forecast = np.maximum(load_forecast, 0)

# Генеруємо добовий патерн цін
daily_energy_price = np.array([price_peak if 17 <= h <= 20 else price_offpeak for h in range(n_hours)])
# Повторюємо добовий патерн цін на весь місяць
energy_price = np.tile(daily_energy_price, days)

# Створюємо DataFrame для всього місяця
data_monthly = {'timestamp': timestamp,
                'load_forecast': load_forecast,
                'energy_price': energy_price}
df_monthly = pd.DataFrame(data_monthly)

print(f"Згенеровано DataFrame розміром: {df_monthly.shape}")

# --- Оптимізація та Евристика (на прикладі першого дня) ---
# Для демонстрації роботи алгоритмів візьмемо дані лише за перший день.

print("\n--- Запуск оптимізації та евристики на даних першого дня ---")

# Вибираємо дані за перший день
load_forecast_day1 = df_monthly['load_forecast'].values[:n_hours]
energy_price_day1 = df_monthly['energy_price'].values[:n_hours]

# Параметри гнучкості навантаження (приклад для одного дня)
shiftable_load = 20  # Загальний обсяг навантаження, який можна переміщувати за день


# Функція для обчислення витрат на електроенергію
def calculate_cost(load_profile, energy_price):
    return np.sum(load_profile * energy_price)

# Функція, яку потрібно мінімізувати (витрати на електроенергію)
def objective_function(x, load_forecast, energy_price, shiftable_load_unused):
    # shiftable_load не використовується прямо тут, але передається minimize
    """Обчислює витрати на електроенергію для заданого графіка навантаження."""
    load_profile = load_forecast + x
    # Додамо штраф, якщо навантаження стає від'ємним (хоча bounds мають це обмежувати)
    penalty = np.sum(np.maximum(0, -load_profile)) * 1000 # Великий штраф
    cost = calculate_cost(load_profile, energy_price) + penalty
    return cost

# Обмеження (Constraints) для одного дня
# 1. Загальна сума переміщеного навантаження (змін x) повинна бути близькою до 0
#    Це означає, що скільки навантаження прибрали з одних годин, стільки ж додали в інші.
constraint_sum = ({'type': 'eq', 'fun': lambda x: np.sum(x)})


# 2. Обмеження на переміщення (bounds) для кожної години:
#    Не можна зменшити навантаження більше, ніж воно є.
#    Не можна збільшити/зменшити навантаження на величину, більшу за shiftable_load (умовно, для прикладу).
#    Важливо: оригінальний код мав потенційну помилку в bounds, якщо load_forecast < shiftable_load.
#    Правильніше обмежити зниження -load_forecast[i], а збільшення - деякою розумною межею (напр., shiftable_load).
bounds_day1 = [(-load_forecast_day1[i], shiftable_load) for i in range(n_hours)]


# Початкове значення (всі зсуви = 0)
x0_day1 = np.zeros(n_hours)

# Використовуємо scipy.optimize.minimize для першого дня
start_time_opt = time.time()
result = minimize(objective_function, x0_day1, args=(load_forecast_day1, energy_price_day1, shiftable_load),
                   method='SLSQP', bounds=bounds_day1,
                   constraints=constraint_sum) # Використовуємо тільки constraint_sum, бо bounds покривають інші аспекти
end_time_opt = time.time()

if result.success:
    # Отримуємо оптимальний графік навантаження для першого дня
    optimized_load_profile_day1 = load_forecast_day1 + result.x

    # Обчислюємо витрати до і після оптимізації для першого дня
    original_cost_day1 = calculate_cost(load_forecast_day1, energy_price_day1)
    optimized_cost_day1 = calculate_cost(optimized_load_profile_day1, energy_price_day1)

    print("\nРезультати з використанням нелінійного програмування (для першого дня):")
    print(f"  Початкові витрати: {original_cost_day1:.2f}")
    print(f"  Оптимізовані витрати: {optimized_cost_day1:.2f}")
    print(f"  Зміна (x): сума={np.sum(result.x):.2f}, мін={np.min(result.x):.2f}, макс={np.max(result.x):.2f}")
    print(f"  Час виконання оптимізації: {end_time_opt - start_time_opt:.4f} секунд")
else:
    print("\nОптимізація не знайшла розв'язку.")
    print(result.message)
    original_cost_day1 = calculate_cost(load_forecast_day1, energy_price_day1)
    print(f"  Початкові витрати: {original_cost_day1:.2f}")


# Евристичний підхід (приклад для першого дня)
def heuristic_load_shifting(load_forecast, energy_price, shiftable_load):
    """Переміщує навантаження з пікових годин в непікові."""
    load_profile = load_forecast.copy()
    peak_hours_indices = np.where(energy_price == np.max(energy_price))[0]
    offpeak_hours_indices = np.where(energy_price == np.min(energy_price))[0]

    if len(peak_hours_indices) == 0 or len(offpeak_hours_indices) == 0:
        print("Не знайдено пікових або непікових годин для евристики.")
        return load_profile # Повертаємо без змін, якщо немає куди переміщати

    # Розраховуємо, скільки можна зняти з кожної пікової години
    total_load_in_peak = np.sum(load_profile[peak_hours_indices])
    load_to_shift_total = min(shiftable_load, total_load_in_peak) # Не можемо зняти більше, ніж є або дозволено

    shifted_amount = 0
    # Знімаємо навантаження з пікових годин пропорційно
    for idx in peak_hours_indices:
        reduction = min(load_profile[idx], load_to_shift_total * (load_profile[idx] / total_load_in_peak if total_load_in_peak > 0 else 0))
        # Перевірка, чи не перевищуємо загальний ліміт
        if shifted_amount + reduction > load_to_shift_total:
            reduction = load_to_shift_total - shifted_amount
        if reduction < 0: reduction = 0 # Уникнення від'ємного зняття

        load_profile[idx] -= reduction
        shifted_amount += reduction
        if shifted_amount >= load_to_shift_total:
            break # Досягли ліміту

    # Розподіляємо зняте навантаження рівномірно по непікових годинах
    load_per_offpeak_hour = shifted_amount / len(offpeak_hours_indices) if len(offpeak_hours_indices) > 0 else 0
    for idx in offpeak_hours_indices:
        load_profile[idx] += load_per_offpeak_hour

    return load_profile

start_time_heur = time.time()
heuristic_load_profile_day1 = heuristic_load_shifting(load_forecast_day1, energy_price_day1, shiftable_load)
end_time_heur = time.time()

heuristic_cost_day1 = calculate_cost(heuristic_load_profile_day1, energy_price_day1)

print("\nРезультати з використанням евристичного підходу (для першого дня):")
print(f"  Евристичні витрати: {heuristic_cost_day1:.2f}")
# Перевірка балансу для евристики
print(f"  Зміна: сума={np.sum(heuristic_load_profile_day1 - load_forecast_day1):.2f}")
print(f"  Час виконання евристики: {end_time_heur - start_time_heur:.6f} секунд")

# Можна також додати візуалізацію для першого дня, якщо потрібно
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(range(n_hours), load_forecast_day1, label='Початкове навантаження', color='blue', linestyle='--')
if result.success:
    plt.plot(range(n_hours), optimized_load_profile_day1, label='Оптимізоване навантаження (SLSQP)', color='red')
plt.plot(range(n_hours), heuristic_load_profile_day1, label='Евристичне навантаження', color='green', linestyle=':')
plt.ylabel('Навантаження (Load)')
plt.xlabel('Година дня')
plt.xticks(range(0, n_hours, 2))
plt.legend()
plt.grid(True)
plt.twinx()
plt.plot(range(n_hours), energy_price_day1, label='Ціна', color='orange', drawstyle='steps-post')
plt.ylabel('Ціна (€/MWh)', color='orange')
plt.tick_params(axis='y', labelcolor='orange')
plt.legend(loc='upper right')
plt.title('Оптимізація графіка навантаження (Перший день)')
plt.show()