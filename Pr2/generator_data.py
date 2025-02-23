import csv
import os
import random
import multiprocessing

def generate_row(num_columns):
    """Генерує один рядок даних."""
    return [random.randint(1, 100) for _ in range(num_columns)]

def generate_large_csv(filename="D:/Mysor2/3kurs/Coll_proc_data/Pr2/large_data.csv", filesize_gb=1, num_columns=5, chunk_size_mb=64):
    """Генерує великий CSV файл заданого розміру, використовуючи multiprocessing."""

    bytes_per_gb = 1024 * 1024 * 1024
    bytes_per_mb = 1024 * 1024
    target_bytes = filesize_gb * bytes_per_gb
    chunk_bytes = chunk_size_mb * bytes_per_mb
    rows_per_chunk = int(chunk_bytes / (num_columns * 4)) #Приблизна оцінка. Ints займають 4 байти.

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Заголовок
        header = [f'column_{i}' for i in range(num_columns)]
        writer.writerow(header)

        current_bytes = os.path.getsize(filename)
        row_count = 0


        with multiprocessing.Pool() as pool: #Кількість процесів = кількості ядер CPU
            while current_bytes < target_bytes:
                # Генеруємо список рядків паралельно
                rows = pool.starmap(generate_row, [(num_columns,) for _ in range(rows_per_chunk)])
                writer.writerows(rows) #Записуємо всі рядки одразу
                row_count += len(rows)
                current_bytes = os.path.getsize(filename)

                if row_count % 1000000 == 0: #Виводимо інформацію кожні 1000000 рядків
                    print(f"Розмір: {current_bytes / bytes_per_mb:.3f} МБ, рядків: {row_count}")

        print(f"Файл '{filename}' успішно згенеровано. Розмір: {current_bytes / bytes_per_gb:.2f} ГБ, рядків: {row_count}")


if __name__ == "__main__":
    generate_large_csv()
