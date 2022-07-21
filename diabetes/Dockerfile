FROM python:3.9.13-slim
WORKDIR /usr/src/diabetes

COPY requirements.txt ./
RUN pip install --no-cache-dir -U -r requirements.txt

COPY . .
 
CMD ["python", "./train.py"]
