# risk-pmml-builder

Builds PMML models for credit card risk scoring.

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