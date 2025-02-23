import pandas as pd
import sqlite3
import time
import multiprocessing

def create_index(db_filename, table_name, index_column):
    """Створює індекс для вказаної таблиці та стовпця."""
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    try:
        cursor.execute(f"CREATE INDEX IF NOT EXISTS {table_name}_{index_column}_idx ON {table_name}({index_column})")
        conn.commit()
    except Exception as e:
        print(f"Помилка при створенні індексу: {e}")
    finally:
        conn.close()

def csv_to_sqlite(csv_filename, db_filename="D:/Mysor2/3kurs/Coll_proc_data/Pr2/database.db", table_name="large_table", index_columns=["column_0"]): #Список індексів
    """Записує великий CSV файл в SQLite базу даних і створює індекси паралельно."""

    start_time = time.time()

    # 1. Читання CSV у DataFrame частинами (chunks)
    chunksize = 100000  # Розмір частини
    i = 0
    for chunk in pd.read_csv(csv_filename, chunksize=chunksize):
        conn = sqlite3.connect(db_filename)
        chunk.to_sql(table_name, conn, if_exists='append', index=False)
        conn.close()
        i += 1
        print(f"Записано частину {i} до бази даних.")

    end_import_time = time.time()
    print(f"Завантаження CSV в SQLite завершено. Час: {end_import_time - start_time:.2f} секунд")

    # 2. Паралельне створення індексів
    start_index_time = time.time()
    with multiprocessing.Pool() as pool:
        pool.starmap(create_index, [(db_filename, table_name, col) for col in index_columns])

    end_index_time = time.time()
    print(f"Індекси створено. Час: {end_index_time - start_index_time:.2f} секунд")

    print(f"Успішно перетворено CSV '{csv_filename}' на SQLite базу даних '{db_filename}' з індексами на колонках '{', '.join(index_columns)}'.")

if __name__ == "__main__":
    csv_to_sqlite("D:/Mysor2/3kurs/Coll_proc_data/Pr2/large_data.csv", index_columns=["column_0", "column_1"]) #Приклад з кількома індексами