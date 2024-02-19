#use official python runtime as parent image 
FROM python:3.8-slim

#set working directory to /app
WORKDIR /docker_app

#xopy the required files and directory into the container at /app
COPY app.py /docker_app/app.py
COPY model.joblib /docker_app/model.joblib
COPY requirements.txt /docker_app/requirements.txt
COPY src/ /docker_app/src/

#install any needed packages specified in requirements.txt file
RUN pip install -r requirements.txt

EXPOSE 8080

#Run app.py when container launches
CMD ["python", "app.py"]