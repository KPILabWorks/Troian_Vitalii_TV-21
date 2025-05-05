import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
try:
    df = pd.read_csv('D:/Mysor2/3kurs/Coll_proc_data/Pr6/Comma.csv', sep=',') # або sep=';'
except Exception as e:
    print(f"Помилка читання CSV: {e}")
    print("Спробуйте перевірити роздільник (sep=',' або sep=';')")
    print("Також перевірте, чи немає зайвих рядків на початку файлу (спробуйте додати skiprows=N)")
    exit()

print("Назви колонок:", df.columns)

column_mapping = {
    df.columns[0]: 'Time',
    df.columns[1]: 'AccX',
    df.columns[2]: 'AccY',
    df.columns[3]: 'AccZ'
}
df = df.rename(columns=column_mapping)

df = df.dropna()

for col in ['Time', 'AccX', 'AccY', 'AccZ']:
     if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce') 
df = df.dropna() 

print("\nПерші 5 рядків даних після перейменування та очистки:")
print(df.head())
print("\nТипи даних:")
print(df.info())

# --- Візуалізація вихідних даних ---
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['AccX'], label='AccX')
plt.plot(df['Time'], df['AccY'], label='AccY')
plt.plot(df['Time'], df['AccZ'], label='AccZ')
plt.title('Вихідні дані акселерометра')
plt.xlabel('Час (с)')
plt.ylabel('Прискорення (м/с^2)')
plt.legend()
plt.grid(True)
plt.show()

# --- Видалення шумів та підготовка до аналізу ---

# Розрахунок загальної амплітуди прискорення 
df['AccMagnitude'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)

# Визначення осі з найбільшими коливаннями під час ходьби 
signal_for_steps = df['AccMagnitude'] 

# Фільтрація сигналу для згладжування шуму (низькочастотний фільтр Баттерворта)
fs = 1 / (df['Time'].iloc[1] - df['Time'].iloc[0]) # Частота дискретизації 
cutoff = 2.0  # Частота зрізу (Гц) - підбирається експериментально (2-5 Гц для ходьби)
order = 4     # Порядок фільтра

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Перевірка наявності даних перед фільтрацією
if len(signal_for_steps) > order * 3:
    filtered_signal = butter_lowpass_filter(signal_for_steps, cutoff, fs, order)
else:
    print("Недостатньо даних для фільтрації.")
    filtered_signal = signal_for_steps.values 

df['FilteredAccMagnitude'] = filtered_signal

# Візуалізація відфільтрованого сигналу
plt.figure(figsize=(12, 6))
plt.plot(df['Time'], signal_for_steps, label='Вихідний сигнал (Магнітуда)', alpha=0.5)
plt.plot(df['Time'], df['FilteredAccMagnitude'], label='Відфільтрований сигнал', linewidth=2)
plt.title('Сигнал прискорення до і після фільтрації')
plt.xlabel('Час (с)')
plt.ylabel('Прискорення (м/с^2)')
plt.legend()
plt.grid(True)
plt.show()

# --- Виявлення кроків (піків) ---
# Параметри для find_peaks (потрібно підібрати!)
# height: мінімальна висота піку (залежить від амплітуди сигналу)
# distance: мінімальна відстань між піками (в семплах) - щоб не детектувати шум як кроки
min_peak_height = np.mean(df['FilteredAccMagnitude'])
min_peak_distance = fs / 3 

peaks, properties = find_peaks(df['FilteredAccMagnitude'], height=min_peak_height, distance=min_peak_distance)

# Перевірка, чи знайдено піки
if len(peaks) == 0:
    print("\nНе вдалося знайти піки (кроки). Спробуйте змінити параметри:")
    print(f"- Поточна мінімальна висота (height): {min_peak_height:.2f}")
    print(f"- Поточна мінімальна відстань (distance): {min_peak_distance:.2f} семплів")
else:
    print(f"\nЗнайдено {len(peaks)} піків (кроків).")

    # Візуалізація піків на відфільтрованому сигналі
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], df['FilteredAccMagnitude'], label='Відфільтрований сигнал')
    plt.plot(df['Time'].iloc[peaks], df['FilteredAccMagnitude'].iloc[peaks], "x", label='Виявлені кроки', markersize=8, color='red')
    plt.title('Виявлення кроків')
    plt.xlabel('Час (с)')
    plt.ylabel('Відфільтроване прискорення (м/с^2)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Розрахунок частоти кроків (каденсу) ---

    first_step_time = df['Time'].iloc[peaks[0]]
    last_step_time = df['Time'].iloc[peaks[-1]]
    walking_duration = last_step_time - first_step_time

    if walking_duration > 0:
        num_steps = len(peaks) - 1 
        step_frequency_hz = num_steps / walking_duration # Частота в Гц (кроків/секунду)
        step_frequency_spm = step_frequency_hz * 60    # Частота в SPM (кроків/хвилину)

        print(f"\nАналіз частоти кроків:")
        print(f"- Тривалість ходьби (між першим і останнім кроком): {walking_duration:.2f} с")
        print(f"- Кількість виявлених кроків за цей період: {len(peaks)}") 
        print(f"- Середня частота кроків (каденс): {step_frequency_hz:.2f} Гц")
        print(f"- Середня частота кроків (каденс): {step_frequency_spm:.2f} кроків/хвилину")
    else:
        print("\nНеможливо розрахувати частоту: виявлено менше 2 кроків або тривалість нульова.")

# --- Оцінка довжини кроку  ---
# Приклад розрахунку середньої амплітуди коливань
avg_peak_amplitude = np.mean(properties['peak_heights'])
print(f"\nСередня амплітуда піків: {avg_peak_amplitude:.2f} м/с^2")
print("\n--- Оцінка довжини кроку ---")

if 'step_frequency_hz' in locals() and 'avg_peak_amplitude' in locals() and 'walking_duration' in locals() and walking_duration > 0:

    # --- Модель 1: На основі амплітуди коливань  ---
    # Припущення: більша амплітуда вертикальних коливань відповідає довшому кроку.
    # C_amp - коефіцієнт, що пов'язує амплітуду піків (м/с^2) і довжину (м).
    C_amp = 0.35 #  (0.3 - 0.5)
    try:
        estimated_length_amp = C_amp * np.sqrt(avg_peak_amplitude)
        print(f"\nМодель 1 (на основі амплітуди коливань):")
        print(f"  - Середня амплітуда піків: {avg_peak_amplitude:.2f} м/с^2")
        print(f"  - Оціночна довжина кроку: {estimated_length_amp:.2f} м (використана константа C_amp={C_amp})")
    except Exception as e:
        print(f"\nПомилка розрахунку Моделі 1: {e}")


    # --- Модель 2 (варіант): Спрощена модель Weinberg (на основі амплітуди) ---
    # Використовує амплітуду коливань. Формула: Length = K * (AvgPeakAmplitude)^(1/4)
    # K_weinberg - константа. Оскільки зріст не використовується, беремо фіксовану.
    K_weinberg = 0.55 
    try:
        estimated_length_weinberg = K_weinberg * (avg_peak_amplitude ** 0.25)
        print(f"\nМодель 2 (спрощена модель Weinberg - амплітуда):") 
        print(f"  - Середня амплітуда піків: {avg_peak_amplitude:.2f} м/с^2")
        print(f"  - Використана константа K_weinberg={K_weinberg:.2f}")
        print(f"  - Оціночна довжина кроку: {estimated_length_weinberg:.2f} м")
    except Exception as e:
        print(f"\nПомилка розрахунку Моделі 2 (Weinberg): {e}")

    if 'step_frequency_spm' in locals():
        print(f"\nДодатково: Розрахована середня частота кроків (каденс): {step_frequency_spm:.2f} кроків/хвилину")

else:
    print("\nНеможливо оцінити довжину кроку: відсутні дані про частоту або амплітуду кроків.")
    if 'walking_duration' in locals() and walking_duration <= 0:
        print("  (Причина: виявлено менше 2 кроків або тривалість ходьби нульова)")


# --- Аналіз даних із використанням алгоритмів машинного навчання  ---

# Створимо мітки для даних: 'standing' або 'walking'
standing_intervals = standing_intervals = [(0, 13.4), (72, 83)]# comma #[(0, 14.2), (74.7, 86)]# comma 100 #  
walking_intervals =   walking_intervals = [(13.4, 72)]          # comma # [(14.2, 74.7)]          # comma 100 #
#
df['Activity'] = 'unknown' # Початкова мітка

for start, end in standing_intervals:
    df.loc[(df['Time'] >= start) & (df['Time'] < end), 'Activity'] = 'standing'
for start, end in walking_intervals:
    df.loc[(df['Time'] >= start) & (df['Time'] < end), 'Activity'] = 'walking'

# Видаляємо нерозмічені дані
df_labeled = df[df['Activity'] != 'unknown'].copy()

if df_labeled.empty:
    print("\nНемає розмічених даних для тренування ML моделі. Перевірте інтервали.")
else:
    print(f"\nПідготовка до ML: знайдено {len(df_labeled)} розмічених записів.")
    print(df_labeled['Activity'].value_counts())

    # --- Створення ознак (Features) для ML ---
    window_size_sec = 1.0 
    window_size_samples = int(window_size_sec * fs)
    step = window_size_samples // 2 

    features = []
    labels = []

    feature_cols = ['AccX', 'AccY', 'AccZ', 'AccMagnitude', 'FilteredAccMagnitude']
    feature_cols = [col for col in feature_cols if col in df_labeled.columns]

    if not feature_cols:
        print("Не знайдено колонок для створення ознак ML.")
    else:
        print(f"\nСтворення ознак для ML з колонок: {feature_cols}")
        for i in range(0, len(df_labeled) - window_size_samples, step):
            window = df_labeled.iloc[i : i + window_size_samples]
            window_features = []
            for col in feature_cols:
                window_features.append(window[col].mean())
                window_features.append(window[col].std())
                window_features.append(window[col].min())
                window_features.append(window[col].max())
                window_features.append(window[col].var()) # Дисперсія

            features.append(window_features)
            labels.append(window['Activity'].mode()[0]) # Найчастіша мітка у вікні

        X = np.array(features)
        y = np.array(labels)

        if len(X) == 0:
             print("Не вдалося створити ознаки для ML. Можливо, вікно занадто велике або даних мало.")
        else:
            print(f"Створено {len(X)} вікон для ML.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

            print(f"\nРозмір тренувальної вибірки: {X_train.shape[0]}")
            print(f"Розмір тестової вибірки: {X_test.shape[0]}")

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # --- Оцінка якості моделі ---
            y_pred = model.predict(X_test)

            print("\n--- Результати ML Класифікації (Ходьба/Стояння) ---")
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Точність моделі (Accuracy): {accuracy:.4f}")
            print("\nЗвіт по класифікації:")
            print(classification_report(y_test, y_pred))
