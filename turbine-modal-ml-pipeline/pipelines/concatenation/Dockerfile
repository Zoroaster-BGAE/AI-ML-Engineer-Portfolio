# 1. Use a minimal, official Python image
FROM python:3.11-slim

# 2. Ensure Python outputs logs immediately
ENV PYTHONUNBUFFERED=1

# 3. Set working directory inside the container
WORKDIR /app

# 4. Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# 5. Install Python dependencies without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy only the source code (everything else is excluded by .dockerignore)
COPY source/ ./source/

# 7. (Optional) Expose ports if your app serves HTTP
# EXPOSE 8000

# 8. Default entrypoint runs the main pipeline script
ENTRYPOINT ["python", "source/main.py"]
