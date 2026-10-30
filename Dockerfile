
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Avoid unnecessary Python cache files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Streamlit uses port 8501
EXPOSE 8501

# Start the PaperMind application
CMD ["streamlit", "run", "PaperMind_Streamlit.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
