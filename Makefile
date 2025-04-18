install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black library/*.py

lint:
	pylint --disable=R,C library/*.py

all: install lint