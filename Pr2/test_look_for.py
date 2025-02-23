import sqlite3
import timeit
import random

def execute_query(db_filename, query):
    """Виконує запит і повертає час виконання."""
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    cursor.execute(query)
    cursor.fetchall()  # Отримуємо всі результати, щоб забезпечити повне виконання запиту
    conn.close()


def measure_query_performance(db_filename, table_name, index_column, num_iterations=10):
    """Вимірює час виконання запитів з індексом і без нього, використовуючи timeit."""

    # Генеруємо випадкове значення для пошуку
    random_value = random.randint(1, 10000000)

    # Запити
    query_with_index = f"SELECT * FROM {table_name} WHERE {index_column} = {random_value}"
    query_without_index = f"SELECT * FROM {table_name} WHERE column_2 = {random_value}"  # Припустимо, що column_2 не індексовано

    # Вимірюємо час виконання за допомогою timeit
    time_with_index = timeit.timeit(lambda: execute_query(db_filename, query_with_index), number=num_iterations)
    time_without_index = timeit.timeit(lambda: execute_query(db_filename, query_without_index), number=num_iterations)


    print(f"Середній час виконання запиту з індексом ({index_column}) (після {num_iterations} ітерацій): {time_with_index / num_iterations:.6f} секунд")
    print(f"Середній час виконання запиту без індексу (column_2) (після {num_iterations} ітерацій): {time_without_index / num_iterations:.6f} секунд")


if __name__ == "__main__":
    db_filename = "D:/Mysor2/3kurs/Coll_proc_data/Pr2/database.db"
    table_name = "large_table"
    index_column = "column_0"  # Змініть на стовпець, який ви індексували

    measure_query_performance(db_filename, table_name, index_column)