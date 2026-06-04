# Импорт библиотек
from modules.context import Context

# Инициализируем приложение
context = Context()

# Запускаем приложение
context.start()

# Вывод в консоль всего текста
print("Итоговый текст:")
for i in context.builder.fragments:
    print(i)
