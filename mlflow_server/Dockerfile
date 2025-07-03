FROM python:3.12-slim

RUN pip install mlflow \
    boto3 \
    pymysql \
    cryptography

EXPOSE 5001

RUN mkdir -p /mlflow
WORKDIR /mlflow