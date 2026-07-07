FROM python:3.11-slim

WORKDIR /app

# Копируем из подпапки telegram_bot в рабочую директорию контейнера
COPY interaction_layer/telegram_bot/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY interaction_layer/telegram_bot/bot.py .
COPY event_bus/ /app/event_bus/

CMD ["python", "bot.py"]
