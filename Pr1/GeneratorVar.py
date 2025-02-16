
def generate_combinations_with_permutations(items):
    """
    Генерує всі можливі комбінації елементів заданого списку (з повтореннями)
    і всі перестановки для кожної комбінації (без itertools),
    видаляючи дублікати.

    Args:
        items: Список елементів.

    Yields:
        Кожен раз список, що представляє комбінацію з перестановкою елементів.
    """

    def generate_permutations(arr):
        #Генерує всі перестановки заданого масиву.
        if len(arr) == 0:
            return [[]]
        permutations = []
        for i in range(len(arr)):
            first = arr[i]
            rest = arr[:i] + arr[i+1:]
            for p in generate_permutations(rest):
                permutations.append([first] + p)
        return permutations

    def generate_combinations(arr, k, start_index=0):
        #Генерує всі комбінації k елементів з заданого масиву (з повтореннями).
        if k == 0:
            return [[]]
        if start_index >= len(arr):
            return []
        combinations = []
        for i in range(start_index, len(arr)):  # Починаємо з start_index
            first = arr[i]
            # Дозволяємо повторне використання елемента, починаючи з того ж індексу
            for c in generate_combinations(arr, k - 1, i):
                combinations.append([first] + c)
        return combinations

    seen = set()  # Множина для зберігання унікальних перестановок (у вигляді кортежів)

    for k in range(len(items) + 1):  # Для кожної довжини комбінації
        for combination in generate_combinations(items, k): #Генеруємо всі комбінації довжини k
            for permutation in generate_permutations(combination): # Генеруємо всі перестановки цієї комбінації
                permutation_tuple = tuple(permutation)  # Перетворюємо список на кортеж
                if permutation_tuple not in seen:  # Перевіряємо, чи немає вже такої перестановки
                    seen.add(permutation_tuple)  # Додаємо перестановку до множини
                    yield list(permutation)  # Видаємо перестановку (у вигляді списку)

# Приклад використання
my_list = [1, 2, 3]
amount = 1
for combination in generate_combinations_with_permutations(my_list):
    print(amount, combination)
    amount += 1