#date: 2023-04-04T16:39:46Z
#url: https://api.github.com/gists/89d4165650a1ac437f36c36e576a4b6b
#owner: https://api.github.com/users/Angelina91

FROM python:3.7-slim

RUN mkdir /app

COPY requirements.txt /app

RUN pip3 install -r /app/requirements.txt --no-cache-dir

#COPY . .

COPY api_yamdb/ app
 
WORKDIR /app

CMD ["gunicorn", "api_yamdb.wsgi:application", "--bind", "0:8000" ] 
