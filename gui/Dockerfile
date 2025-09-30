FROM python:3.11-slim

WORKDIR /app

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer puerto de Streamlit
EXPOSE 8501

# Comando para correr Streamlit
CMD ["streamlit", "run", "gui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]