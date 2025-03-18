#Головне працююче
import featuretools as ft
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import random

# 1. Генеруємо синтетичні дані (без часових міток)
def generate_synthetic_data(n_customers=100, n_readings=1000, n_products=5):
    """Генерує синтетичні дані про енергоспоживання."""
    customer_ids = range(1, n_customers + 1)
    product_ids = range(1, n_products + 1)

    data = []
    for customer_id in customer_ids:
        for reading_number in range(n_readings):
            product_id = random.choice(product_ids)
            energy_consumption = np.random.normal(loc=100 + product_id * 10 + customer_id * 0.5, scale=20) #Залежність споживання від ID
            data.append([customer_id, product_id, energy_consumption])

    df = pd.DataFrame(data, columns=['customer_id', 'product_id', 'energy_consumption'])
    return df

# Згенеруємо невеликий обсяг даних.
synthetic_data = generate_synthetic_data(n_customers=10, n_readings=100)
print(synthetic_data.head())
synthetic_data["reading_id"] = range(len(synthetic_data)) #Додаємо поле index.

# 2. Створюємо EntitySet
es = ft.EntitySet(id="energy_data")

# Додавання сутності "readings"
es = es.add_dataframe(
    dataframe_name="readings",
    dataframe=synthetic_data,
    index="reading_id",
)

# Створення сутності "customers"
customers_df = synthetic_data[["customer_id"]].drop_duplicates()
es = es.add_dataframe(
    dataframe_name="customers",
    dataframe=customers_df,
    index="customer_id",
)

# Створення сутності "products"
products_df = synthetic_data[["product_id"]].drop_duplicates()
es = es.add_dataframe(
    dataframe_name="products",
    dataframe=products_df,
    index="product_id",
)

# Додавання відносин
es = es.add_relationship("customers", "customer_id", "readings", "customer_id")
es = es.add_relationship("products", "product_id", "readings", "product_id")


# 3. Генеруємо ознаки вручну
# Обчислюємо середнє споживання енергії для кожного клієнта
mean_consumption = synthetic_data.groupby('customer_id')['energy_consumption'].mean().reset_index()
mean_consumption.rename(columns={'energy_consumption': 'mean_energy_consumption'}, inplace=True)

# Обчислюємо максимальне споживання енергії для кожного продукту
max_consumption = synthetic_data.groupby('product_id')['energy_consumption'].max().reset_index()
max_consumption.rename(columns={'energy_consumption': 'max_energy_consumption'}, inplace=True)

# Об'єднуємо згенеровані ознаки з основним DataFrame
feature_matrix = pd.merge(synthetic_data, mean_consumption, on='customer_id', how='left')
feature_matrix = pd.merge(feature_matrix, max_consumption, on='product_id', how='left')

# 4. Deep Feature Synthesis (DFS) - Тільки для структурних ознак
target_dataframe_name = "readings"

#Обмежуємось тільки identity примітивами
feature_matrix_2, features_defs = ft.dfs(entityset=es,
                                           target_dataframe_name=target_dataframe_name,
                                           agg_primitives=[],
                                           trans_primitives=[], #Тільки identity
                                           )

#Видаляємо reading_id, якщо він існує (може не бути, якщо DFS не створив жодних ознак)
if 'reading_id' in feature_matrix_2.columns:
    feature_matrix_2 = feature_matrix_2.drop('reading_id', axis=1)

#Обєднуємо результати
feature_matrix = feature_matrix.join(feature_matrix_2, how='left', lsuffix='_manual', rsuffix='_ft')

#5.Очистка
feature_matrix = feature_matrix.loc[:,~feature_matrix.columns.duplicated(keep='first')].copy() #Видаляємо дублікати, залишаємо перші

# 6. Підготовка даних для машинного навчання
#Визначаємо назви стовпців
features = ['customer_id_manual', 'product_id_manual', 'mean_energy_consumption', 'max_energy_consumption']
features = [f for f in features if f in feature_matrix.columns] #Перевіряємо чи є стовпці
labels = 'energy_consumption_manual' #Використовуємо _manual

X = feature_matrix[features]
y = feature_matrix[labels]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Навчання моделі
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 8. Оцінка моделі
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"RMSE: {rmse:.2f}")