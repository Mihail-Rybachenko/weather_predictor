import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from datetime import datetime

# Константы
EPOCHS = 150  # Количество эпох для обучения
BATCH_SIZE = 10  # Размер батча


def load_and_prepare_data(filename):
    """
    Загружает данные из файла и подготавливает их для обучения и тестирования.
    Ожидается файл с числовыми данными: строки — годы, столбцы — 12 месяцев.
    """
    if not os.path.exists(filename):
        print(f"Ошибка: файл {filename} не найден.")
        return None, None, None, None, None

    try:
        data = np.loadtxt(filename, delimiter=',')
        print(f"Данные загружены из {filename}. Размер: {data.shape}")
    except ValueError:
        print("Ошибка: неверный формат данных в файле. Ожидаются числа, разделённые запятыми.")
        return None, None, None, None, None

    if data.shape[1] != 12:
        print(f"Ошибка: ожидалось 12 столбцов (месяцев), получено {data.shape[1]}.")
        return None, None, None, None, None

    # Разделяем данные: X — входные данные, y — целевые значения (сдвиг на 1 год)
    X, y = data[:-1], data[1:]
    train_size = len(X) - 2  # Оставляем 2 года для тестовой выборки
    print(f"Разделение данных: {train_size} строк для обучения, {len(X) - train_size} для теста.")
    return X, X[:train_size], X[train_size:], y[:train_size], y[train_size:]


def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Создаёт и обучает нейронную сеть для предсказания температуры по месяцам.
    """
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Входной слой
        Dense(32, activation='relu'),  # Скрытый слой
        Dense(12)  # Выходной слой: 12 месяцев
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    print("Обучение модели началось...")

    history = model.fit(X_train, y_train,
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        validation_split=0.2, verbose=1)

    loss, mae = model.evaluate(X_test, y_test)
    print(f"Оценка на тестовых данных: Loss = {loss:.4f}, MAE = {mae:.4f}")
    return model, history


def show_training_history(history):
    """
    Визуализирует процесс обучения модели: MAE на обучающей и валидационной выборках.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='MAE на обучении')
    plt.plot(history.history['val_mae'], label='MAE на валидации')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (MAE)')
    plt.title('Динамика обучения модели')
    plt.legend()
    plt.grid()
    plt.show()
    print("График обучения построен.")


def make_prediction(model, X):
    """
    Выполняет предсказание температуры на следующий год на основе последних данных.
    Сохраняет результат в файл и строит график.
    """
    print("Генерация предсказания для следующего года...")
    prediction = model.predict(X[-1:])[0]  # Предсказание для последней строки

    # Вывод предсказания по месяцам
    months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    print("Предсказанные температуры:")
    for month, temp in zip(months, prediction):
        print(f"{month}: {temp:.2f}°C")

    # Сохранение результата
    np.savetxt('prediction_next_year.txt', prediction, delimiter=',')
    print("Предсказание сохранено в 'prediction_next_year.txt'.")

    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(months, prediction, marker='o', color='b')
    plt.xlabel('Месяц')
    plt.ylabel('Температура (°C)')
    plt.title('Прогноз температуры на 2025 год')
    plt.grid()
    plt.show()
    print("График предсказания построен.")


def save_model(model):
    """
    Сохраняет обученную модель в файл с временной меткой.
    """
    filename = f"weather_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    model.save(filename)
    print(f"Модель сохранена в файл: {filename}")


def main():
    """
    Основная функция программы. Управляет меню и логикой работы с моделью и данными.
    """
    tf.random.set_seed(42)  # Фиксируем генератор случайных чисел для воспроизводимости
    np.random.seed(42)

    X, X_train, X_test, y_train, y_test = None, None, None, None, None
    model, history = None, None

    while True:
        print("\n=== Меню программы ===")
        print("1. Загрузить данные из файла")
        print("2. Обучить модель")
        print("3. Загрузить готовую модель")
        print("4. Сделать предсказание")
        print("5. Показать график обучения")
        print("6. Сохранить модель")
        print("7. Выход")
        print("====================")
        choice = input("Выберите действие (1-7): ")

        if choice == '1':
            file_path = input("Введите путь к файлу с данными(изначально data/temperatures_2014_2024.txt): ")
            X, X_train, X_test, y_train, y_test = load_and_prepare_data(file_path)
        elif choice == '2':
            if X_train is None:
                print("Ошибка: сначала загрузите данные.")
            else:
                model, history = create_and_train_model(X_train, y_train, X_test, y_test)
        elif choice == '3':
            model_path = input("Введите путь к файлу модели: ")
            try:
                model = tf.keras.models.load_model(model_path)
                print("Модель успешно загружена.")
            except Exception as e:
                print(f"Ошибка при загрузке модели: {e}")
        elif choice == '4':
            if model is None or X is None:
                print("Ошибка: необходимы данные и модель для предсказания.")
            else:
                make_prediction(model, X)
        elif choice == '5':
            if history is None:
                print("Ошибка: сначала обучите модель.")
            else:
                show_training_history(history)
        elif choice == '6':
            if model is None:
                print("Ошибка: нет модели для сохранения.")
            else:
                save_model(model)
        elif choice == '7':
            print("Завершение работы программы.")
            break
        else:
            print("Неверный выбор. Пожалуйста, введите число от 1 до 7.")


if __name__ == "__main__":
    main()