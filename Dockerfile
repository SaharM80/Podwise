# Base Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip uvicorn

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies from pyproject.toml directly
RUN pip install .

# Copy the app code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with your .env
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--env-file", ".env"]
