FROM python:3.12-slim

WORKDIR /code

# Installer les dépendances d'abord (profite du cache Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code de l'API et le modèle entraîné
COPY app/ app/
COPY models/ models/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]