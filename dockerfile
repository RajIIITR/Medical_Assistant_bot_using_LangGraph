# base image
FROM python:3.10-slim-buster

# set work directory
WORKDIR /app

# copy
COPY . /app 

# install dependencies
RUN pip install -r requirements.txt

# PORT
EXPOSE 8080

# command
CMD ["python", "./app.py"]