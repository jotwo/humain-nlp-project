FROM ubuntu:latest

# Install reqs
RUN apt-get update -y \
	&& apt-get install -y python3-dev python3-pip \
	&& rm -rf /var/lib/apt/lists/*

# integrate only necessary files
RUN mkdir app
RUN mkdir app/templates

COPY src/templates /app/templates
COPY src/*.py /app/
COPY src/Procfile /app
COPY docker_requirements.txt /app

WORKDIR /app
RUN pip3 install --upgrade pip
RUN pip3 install -r docker_requirements.txt

# files paths in py scripts will be parsed relative to WORKDIR
WORKDIR /

# library/module imports are relative to PYTHONPATH
# (defaults to app.py location but overwrite it with workdir? here)
CMD export PYTHONPATH="${PYTHONPATH}:/" ; python3 app/app.py

# CMD python3 app/app.py