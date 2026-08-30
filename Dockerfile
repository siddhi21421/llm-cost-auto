FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the classifier at build time so the image is ready to serve traffic
# the moment it starts (no cold-start training step in the API container).
RUN python -m app.classifier.train

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
