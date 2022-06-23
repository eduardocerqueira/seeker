#date: 2022-06-23T17:10:40Z
#url: https://api.github.com/gists/fb08a65504dec064ccc9967c3ee3e585
#owner: https://api.github.com/users/paulasantanazup

FROM python:3.6
RUN pip install flask
COPY . /opt/
EXPOSE 8080
WORKDIR /opt
ENTRYPOINT ["python", "app.py"]