FROM python:3.9.13-slim
WORKDIR /diabetes

COPY requirements.txt /diabetes/
RUN pip install --no-cache-dir -U -r requirements.txt

COPY . /diabetes/

CMD ["python", "/diabetes/train.py"]
