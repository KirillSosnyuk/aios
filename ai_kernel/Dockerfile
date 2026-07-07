FROM python:3.11-slim

WORKDIR /app

# Копируем из подпапки ai_kernel в рабочую директорию контейнера
COPY ai_kernel/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_kernel/kernel.py .
COPY event_bus/ /app/event_bus/

CMD ["python", "kernel.py"]
