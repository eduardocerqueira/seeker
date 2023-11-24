#date: 2023-11-24T16:45:32Z
#url: https://api.github.com/gists/d0f1f0eff4a4271c71196a572f721d78
#owner: https://api.github.com/users/otrack

FROM python:3.10

WORKDIR /app
RUN pip install delegator.py
RUN pip install web.py
RUN apt-get update && apt-get install -y imagemagick
RUN echo "<policy domain=\"coder\" rights=\"read|write\" pattern=\"PDF\" />" >> /etc/ImageMagick-6/policy.xml
ADD pdfmagic.py /app
EXPOSE 8080

CMD ["python", "/app/pdfmagic.py"]
