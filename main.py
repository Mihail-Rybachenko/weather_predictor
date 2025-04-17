import os
# Модуль для работы с файловой системой
import numpy as np
# Библиотека для работы с массивами
import tensorflow as tf
# Библиотека для создания и обучения нейронной сети
from tensorflow.keras.models import Sequential
# Класс для создания последовательной модели нейросети
from tensorflow.keras.layers import Dense
# Класс для создания полносвязных слоёв
import matplotlib.pyplot as plt
# Библиотека для построения графиков
from datetime import datetime
# Модуль для работы с датой и временем
import tkinter as tk
# Модуль для создания графического интерфейса
from tkinter import ttk, scrolledtext, messagebox
# Дополнительные виджеты tkinter для интерфейса

# Константы
EPOCHS = 150  # Количество эпох для обучения
BATCH_SIZE = 10  # Размер батча

# Глобальные переменные приложения
X, X_train, X_test, y_train, y_test = None, None, None, None, None
model, history = None, None
default_file_path = "data/temperatures_2014_2024.txt"
root = None
file_path_entry = None
model_path_entry = None
year_entry = None
output_text = None
progress_bar = None  # Новая переменная для прогресс-бара

def print_message(message):
    """
    Выводит сообщение в текстовое поле.
    """
    output_text.insert(tk.END, message + "\n")
    output_text.see(tk.END)

def show_error(message):
    """
    Показывает всплывающее окно с ошибкой.
    """
    messagebox.showerror("Ошибка", message)
    print_message(f"Ошибка: {message}")

def show_info(message):
    """
    Показывает всплывающее окно с информацией.
    """
    messagebox.showinfo("Уведомление", message)
    print_message(message)

def load_my_weather_data(filename):
    """
    Загружает данные из файла и подготавливает их для обучения и тестирования.
    Ожидается файл с числовыми данными: строки — годы, столбцы — 12 месяцев.
    """
    if not os.path.exists(filename):
        return None, None, None, None, None
    try:
        data = np.loadtxt(filename, delimiter=',')
    except ValueError:
        return None, None, None, None, None
    if data.shape[1] != 12:
        return None, None, None, None, None
    X, y = data[:-1], data[1:]
    train_size = len(X) - 2
    return X, X[:train_size], X[train_size:], y[:train_size], y[train_size:]

def create_and_train_model(X_train, y_train, X_test, y_test):
    """
    Создаёт и обучает нейронную сеть для предсказания температуры по месяцам.
    Обновляет прогресс-бар во время обучения.
    """
    global progress_bar, root

    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(12)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Callback для обновления прогресс-бара
    class ProgressBarCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Вычисляем процент завершения
            progress = (epoch + 1) / EPOCHS * 100
            progress_bar['value'] = progress
            root.update()  # Обновляем интерфейс

    # Обучаем модель с callback
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=0,  # Отключаем стандартный вывод в консоль
        callbacks=[ProgressBarCallback()]
    )
    # Сбрасываем прогресс-бар после обучения
    progress_bar['value'] = 0
    root.update()

    loss, mae = model.evaluate(X_test, y_test)
    return model, history

def show_my_training(history):
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

def make_my_forecast(model, X, target_year):
    """
    Выполняет предсказание температуры на указанный год (2025–2030).
    Возвращает текст предсказания и строит график.
    """
    last_year = 2024
    years_to_predict = target_year - last_year
    if years_to_predict < 1 or years_to_predict > 6:
        return f"Ошибка: выберите год в диапазоне 2025–2030. Введённый год: {target_year}."

    current_input = X[-1:]
    all_predictions = []

    for year in range(last_year + 1, target_year + 1):
        prediction = model.predict(current_input)[0]
        all_predictions.append((year, prediction))
        current_input = prediction.reshape(1, -1)

    months = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
    output = "Предсказанные температуры:\n"
    for year, prediction in all_predictions:
        output += f"\nГод {year}:\n"
        for month, temp in zip(months, prediction):
            output += f"{month}: {temp:.2f}°C\n"

    forecast_file = os.path.abspath('my_weather_forecast.txt')
    try:
        with open(forecast_file, 'w') as f:
            for year, prediction in all_predictions:
                f.write(f"Год {year}:\n")
                for month, temp in zip(months, prediction):
                    f.write(f"{month}: {temp:.2f}°C\n")
                f.write("\n")
        output += f"\nПредсказание сохранено в '{forecast_file}'."
    except Exception as e:
        show_error(f"Не удалось сохранить прогноз в '{forecast_file}': {e}")
        output += f"\nОшибка при сохранении прогноза в '{forecast_file}': {e}"

    plt.figure(figsize=(10, 6))
    plt.plot(months, all_predictions[-1][1], marker='o', color='b')
    plt.xlabel('Месяц')
    plt.ylabel('Температура (°C)')
    plt.title(f'Прогноз температуры на {target_year} год')
    plt.grid()
    plt.show()

    output += "\nГрафик предсказания построен."
    return output

def save_my_model(model):
    """
    Сохраняет обученную модель в файл с временной меткой.
    """
    filename = f"weather_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras"
    absolute_path = os.path.abspath(filename)
    try:
        model.save(absolute_path)
        return (f"Модель успешно сохранена в файл: {absolute_path}\n"
                f"Имя файла: {filename}\n"
                f"Путь: {os.path.dirname(absolute_path)}")
    except Exception as e:
        return (f"Ошибка при сохранении модели в файл: {absolute_path}\n"
                f"Имя файла: {filename}\n"
                f"Путь: {os.path.dirname(absolute_path)}\n"
                f"Ошибка: {e}")

def load_data():
    """
    Загружает данные из файла, указанного в поле ввода.
    """
    global X, X_train, X_test, y_train, y_test
    file_path = file_path_entry.get()
    X, X_train, X_test, y_train, y_test = load_my_weather_data(file_path)
    if X is None:
        show_error(f"Файл {file_path} не найден или неверный формат данных.")
    else:
        show_info(f"Данные загружены из {file_path}. Размер: {X.shape}\n"
                  f"Разделение данных: {len(X_train)} строк для обучения, {len(X_test)} для теста.")

def train_model():
    """
    Обучает модель на загруженных данных.
    """
    global model, history, X_train, y_train, X_test, y_test
    if X_train is None:
        show_error("Сначала загрузите данные.")
        return
    print_message("Обучение модели началось...")
    model, history = create_and_train_model(X_train, y_train, X_test, y_test)
    show_info(f"Оценка на тестовых данных: Loss = {model.evaluate(X_test, y_test)[0]:.4f}, "
              f"MAE = {model.evaluate(X_test, y_test)[1]:.4f}")

def load_model():
    """
    Загружает модель из файла, указанного в поле ввода.
    """
    global model
    model_path = model_path_entry.get()
    try:
        model = tf.keras.models.load_model(model_path)
        show_info(f"Модель успешно загружена из {model_path}.")
    except Exception as e:
        show_error(f"Не удалось загрузить модель из {model_path}: {e}")

def make_forecast():
    """
    Выполняет предсказание на указанный год.
    """
    global model, X
    if model is None or X is None:
        show_error("Необходимы данные и модель для предсказания.")
        return
    try:
        target_year = int(year_entry.get())
        if target_year < 2025 or target_year > 2030:
            show_error(f"Год должен быть в диапазоне 2025–2030. Вы ввели: {target_year}.")
            return
        result = make_my_forecast(model, X, target_year)
        print_message(result)
    except ValueError:
        show_error("Введите корректный год (целое число).")

def show_training():
    """
    Показывает график обучения модели.
    """
    global history
    if history is None:
        show_error("Сначала обучите модель.")
        return
    show_my_training(history)

def save_model():
    """
    Сохраняет текущую модель.
    """
    global model
    if model is None:
        show_error("Нет модели для сохранения.")
        return
    result = save_my_model(model)
    if "успешно" in result.lower():
        show_info(result)
    else:
        show_error(result)

def create_widgets():
    """
    Создаёт элементы интерфейса (кнопки, поля ввода, текстовое поле, прогресс-бар).
    """
    global file_path_entry, model_path_entry, year_entry, output_text, progress_bar

    # Поле для пути к файлу
    ttk.Label(root, text="Путь к файлу с данными:").pack(pady=5)
    file_path_entry = ttk.Entry(root, width=50)
    file_path_entry.insert(0, default_file_path)
    file_path_entry.pack()

    # Кнопка загрузки данных
    ttk.Button(root, text="1. Загрузить данные", command=load_data).pack(pady=5)

    # Прогресс-бар для обучения
    ttk.Label(root, text="Прогресс обучения:").pack(pady=5)
    progress_bar = ttk.Progressbar(root, length=400, mode='determinate')
    progress_bar.pack()

    # Кнопка обучения модели
    ttk.Button(root, text="2. Обучить модель", command=train_model).pack(pady=5)

    # Поле и кнопка для загрузки модели
    ttk.Label(root, text="Путь к файлу модели:").pack(pady=5)
    model_path_entry = ttk.Entry(root, width=50)
    model_path_entry.pack()
    ttk.Button(root, text="3. Загрузить старую модель", command=load_model).pack(pady=5)

    # Поле и кнопка для предсказания
    ttk.Label(root, text="Год для прогноза (2025–2030):").pack(pady=5)
    year_entry = ttk.Entry(root, width=10)
    year_entry.pack()
    ttk.Button(root, text="4. Сделать прогноз", command=make_forecast).pack(pady=5)

    # Кнопка показа графика обучения
    ttk.Button(root, text="5. Показать график обучения", command=show_training).pack(pady=5)

    # Кнопка сохранения модели
    ttk.Button(root, text="6. Сохранить модель", command=save_model).pack(pady=5)

    # Кнопка выхода
    ttk.Button(root, text="7. Выйти", command=root.quit).pack(pady=5)

    # Текстовое поле для вывода сообщений
    output_text = scrolledtext.ScrolledText(root, width=70, height=10)
    output_text.pack(pady=10)

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Weather Buddy — Прогноз температуры")
    root.geometry("600x500")
    create_widgets()
    root.mainloop()