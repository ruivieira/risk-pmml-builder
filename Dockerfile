FROM fedora:29

COPY builder.py .
COPY data.py .
COPY requirements.txt .

RUN mkdir ./models

RUN dnf -y update && \
    dnf -y install python3 python3-pip java-1.8.0-openjdk

RUN pip3 install -r requirements.txt

CMD ["python3", "./builder.py"]
