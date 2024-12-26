#date: 2024-12-26T16:34:27Z
#url: https://api.github.com/gists/339f4936dcdd600f985b293a523bf429
#owner: https://api.github.com/users/Bitsy-Chuck

#FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:emulators
FROM debian:latest
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    vim \
    procps \
    net-tools \
    iputils-ping \
    lsof \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default python
RUN ln -s /usr/bin/python3 /usr/bin/python

EXPOSE 1000

ENV CLOUDSDK_CORE_DISABLE_PROMPTS 1
ENV HOST_PORT 1000

RUN curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-454.0.0-linux-x86_64.tar.gz
RUN tar -xf google-cloud-cli-454.0.0-linux-x86_64.tar.gz
RUN ./google-cloud-sdk/install.sh
ENV PATH /google-cloud-sdk/bin:$PATH
ENV APP_HOME /work
WORKDIR $APP_HOME

COPY . $APP_HOME

RUN gcloud components update -q
RUN gcloud components update beta -q
RUN gcloud components install -q bigtable beta cbt

# Add killall script
RUN echo '#!/bin/bash' > ./kill-bigtable && \
    echo 'pgrep -f bigtable | xargs -r kill -9' >> ./kill-bigtable && \
    chmod +x kill-bigtable

RUN echo '#!/bin/bash' > ./start-bigtable && \
    echo './kill-bigtable' >> ./start-bigtable && \
    echo 'gcloud beta emulators bigtable start --host-port "0.0.0.0:${HOST_PORT}"' >> ./start-bigtable && \
    chmod +x start-bigtable

RUN echo "Starting Bigtable emulator on port ${HOST_PORT}"

CMD ["./start-bigtable"]