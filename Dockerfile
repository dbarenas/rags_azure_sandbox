# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script into the container at /app
COPY generate_env.py .

# Set environment variables that the script expects (if not passed during runtime)
# ENV AZURE_SUBSCRIPTION_ID YOUR_SUBSCRIPTION_ID_HERE
# Note: It's better to pass AZURE_SUBSCRIPTION_ID at runtime via `docker run -e`

# The command to run when the container launches
CMD ["python", "generate_env.py"]
