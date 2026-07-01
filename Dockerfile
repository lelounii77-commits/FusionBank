FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ai_service.py .
COPY bank_risk_model.pkl .
COPY customers.csv .
COPY accounts.csv .
COPY obligations.csv .

# Expose port (Railway sets PORT env variable)
EXPOSE 5001

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-5001} --workers 2 --timeout 120 ai_service:app
