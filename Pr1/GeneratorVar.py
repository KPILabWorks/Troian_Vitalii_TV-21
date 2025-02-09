def factorial_recursive(n):
  if n < 0:
    raise ValueError("Факторіал не визначений для від'ємних чисел.")
  if n == 0:
    return 1
  else:
    return n * factorial_recursive(n-1)
  
def genVariants(array):
    """Генерує всі можливі перестановки елементів масиву рекурсивно."""
    if len(array) == 0:
        return [[]]  # Повертаємо порожнє

    permutations = []
    for i in range(len(array)):
        first_element = array[i] #Перший елемент
        rest_of_elements = array[:i] + array[i+1:]  # Решта елементів

        # Рекурсивно викликаємо перестановки для решти елементів
        sub_permutations = genVariants(rest_of_elements)

        # Додаємо поточний елемент на початок кожної перестановки решти елементів
        for sub_permutation in sub_permutations:
            print("first_element ", first_element)
            print("sub_permutation ", sub_permutation)
            permutations.append([first_element] + sub_permutation)

    return permutations

example = [1, 2, 3, 4]
result = genVariants(example)
amount = 1
for row in result:
    print(amount,row)
    amount+=1
""" """

"""def factorial_recursive(n):
  if n < 0:
    raise ValueError("Факторіал не визначений для від'ємних чисел.")
  if n == 0:
    return 1
  else:
    return n * factorial_recursive(n-1)
  
def genVariants(array):
    num_row = factorial_recursive(len(array))
    num_cols = len(array)
    my_answear = [[None] * num_cols for _ in range(num_row)] 
    work_num = num_row/num_cols
    val = 0
    y=0

    for i in range(num_row):
      if val == len(array):
        val = 0 
      
      my_answear[i][0] = array[val]
      #print(my_answear[i][0])
      val+=1
    
    for x in range(1,num_cols):
      print(x)
      val = 0
      #for y in range(num_row):
      iterations = 0
      while any(row[x] is None for row in my_answear) and iterations < 1000:
        iterations +=1
        if y >= 24:
          y=0
        if work_num == 0:
          val +=1
          work_num = num_row/num_cols
        temp = array[val]
        if temp in my_answear[x]:   
          pass
        else:
          my_answear[y][x] = temp  
          work_num-=1
          y+=1  
    return my_answear    

example = [1, 2 , 3, 4]
result = genVariants(example)
for row in result:
    print(row)  """    