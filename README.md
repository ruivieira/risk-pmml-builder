# risk-pmml-builder

Builds PMML models for credit card risk scoring.

A PMML representation of a linear regression and a random forest for both
a _card holder risk_ and a _dispute risk_ models are built.

The generated synthetic datasets are saved under `data` and the PMML models
will be saved under `models`. 

## Requirements

* Python 3.x
* JDK (tested with [OpenJDK 8](https://openjdk.java.net/install/))

Alternatively, use Docker (as detailed below).

## Running

Locally, just run:
```bash
$ pip install -r requirements
$ python builder.py 
```

Using docker:
```bash
$ docker build -t risk-pmml-builder:latest .
$ docker run -t risk-pmml-builder:latest
```