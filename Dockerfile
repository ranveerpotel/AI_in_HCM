FROM python:3.11-slim

WORKDIR /app

# System deps for scipy/numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "src.main"]
