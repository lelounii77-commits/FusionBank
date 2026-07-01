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
COPY aisummary.csv .

# Verify all files are present at build time
RUN ls -lh /app/

# Expose port (Railway sets PORT env variable)
EXPOSE 8080

# Run with gunicorn for production
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 ai_service:app
