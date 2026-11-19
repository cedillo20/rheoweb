# SpringPlot: imagen para Railway (sirve también para Render, Fly.io o un VPS)
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias primero: si solo cambia el código, este paso se reutiliza y el build es más rápido
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Código: main.py busca la página en ../frontend
COPY backend backend
COPY frontend frontend

# Usuario sin privilegios
RUN useradd --create-home springplot
USER springplot

WORKDIR /app/backend
EXPOSE 8000
# Railway asigna el puerto en la variable PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
