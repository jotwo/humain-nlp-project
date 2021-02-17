FROM      ubuntu:20.10
COPY      requirements.txt .

# Update the package lists and install python3 and pip3
RUN       apt-get update \
       && apt-get install -y python3 python3-pip

# Install requirements
RUN       pip3 install --upgrade -r requirements.txt
RUN       python3 -m spacy download en_core_web_sm

# Create file system hierarchy
RUN       mkdir /app
RUN       mkdir /app/src
RUN       mkdir /app/uploads

COPY      src /app/src

WORKDIR   /app/src

CMD       ["python3","app.py"]
