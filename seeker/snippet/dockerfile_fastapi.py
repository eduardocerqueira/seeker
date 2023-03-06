#date: 2023-03-06T17:07:15Z
#url: https://api.github.com/gists/c3d53f63df3be5ca6a19c52a1c65a619
#owner: https://api.github.com/users/aruberts

# Start from a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the application code into the container
COPY ["loan_catboost_model.cbm", "app.py", "./"] .

# Expose the app port
EXPOSE 80

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]