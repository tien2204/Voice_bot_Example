FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --retries=3 -r requirements.txt
COPY ./app ./app
EXPOSE 8000
ENV PYTHONDONTWRITEBYTECODE 0
ENV PYTHONUNBUFFERED 0
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
