import pandas as pd
import numpy as np

# 1. Завантаження набору даних та додавання заголовків
filename = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv"

headers = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight", "engine-type",
    "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
    "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"
]

df = pd.read_csv(filename, names=headers)

# Перетворення "?" на NaN
df.replace("?", np.nan, inplace=True)

# Завдання 1: Замінити NaN у стовпці 'stroke' середнім значенням
avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average of stroke:", avg_stroke)
df["stroke"] = df["stroke"].replace(np.nan, avg_stroke).astype("float")

# Завдання 2: Нормалізувати стовпець 'height' (Simple Feature Scaling)
df["height"] = df["height"] / df["height"].max()
print("\nНормалізований стовпець 'height' (перші 5 значень):")
print(df["height"].head())

# Завдання 3.1: Створити змінну індикатора для стовпця 'aspiration'
dummy_variable_2 = pd.get_dummies(df["aspiration"], dtype=int)
dummy_variable_2.rename(
    columns={"std": "aspiration-std", "turbo": "aspiration-turbo"},
    inplace=True
)

# Завдання 3.2: Об'єднати новий фрейм із вихідним та видалити 'aspiration'
df = pd.concat([df, dummy_variable_2], axis=1)
df.drop("aspiration", axis=1, inplace=True)

# Завдання 4: Замінити усі NaN у всіх стовбцях середнім значенням
for col in df.columns:
    if df[col].isnull().sum() > 0:
        try:
            numeric_col = df[col].astype("float")
            mean_val = numeric_col.mean(axis=0)
            df[col] = numeric_col.replace(np.nan, mean_val)
            print(f"Стовпець '{col}': NaN замінено середнім значенням {mean_val:.4f}")
        except ValueError:
            # Категоріальний стовпець 'num-of-doors' заповнюється найчастішим значенням
            most_freq = df[col].value_counts().idxmax()
            df[col] = df[col].replace(np.nan, most_freq)
            print(f"Стовпець '{col}': NaN замінено найчастішим значенням '{most_freq}'")

# Коригування форматів даних
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")

print("\nЗагальна кількість NaN у наборі даних після обробки:", df.isnull().sum().sum())
print("Розмірність підготовленого набору даних:", df.shape)
