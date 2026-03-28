FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code (main.py and index.html)
COPY . .

# Cloud Run uses the PORT environment variable
ENV PORT 8080

# Start the application
CMD ["python", "main.py"]