# Dockerfile
FROM python:3.11

# Create app directory
WORKDIR /app

# Copy all files
COPY . .

# Make sure setup script is executable
RUN chmod +x setup.sh

# Run setup.sh to install dependencies
RUN ./setup.sh

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Command to run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
