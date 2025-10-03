#date: 2025-10-03T16:36:02Z
#url: https://api.github.com/gists/7099812da62c2092a6b59f3b4c8611a7
#owner: https://api.github.com/users/russ

FROM node:gallium-buster as jsbuilder

ARG BUILD_DATE
ARG GIT_REF
ARG GIT_SHA

ENV NODE_OPTIONS=--max_old_space_size=4096

WORKDIR /app

COPY . /app/
RUN yarn && yarn build


FROM 84codes/crystal:master-ubuntu-jammy as builder

ARG BUILD_DATE
ARG GIT_REF
ARG GIT_SHA

WORKDIR /app

# Cache dependencies
COPY ./shard.yml ./shard.lock /app/
RUN shards install --production -v

COPY . /app/
COPY --from=jsbuilder /app/public/*.png /app/public/
COPY --from=jsbuilder /app/public/manifest.json /app/public/manifest.json
COPY --from=jsbuilder /app/public/css /app/public/css
COPY --from=jsbuilder /app/public/js /app/public/js
RUN shards build --static --release --production -DGIT_SHA="${GIT_SHA}" -DGIT_REF="${GIT_REF}" -DBUILD_DATE="${BUILD_DATE}"

# Run the binary (result in one layer)
# FROM tigerdockermediocore/cuda-ffmpeg-docker:5.0-cu116
FROM linuxserver/ffmpeg:latest

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
ENV BUILD_DATE=$BUILD_DATE
ENV GIT_REF=$GIT_REF
ENV GIT_SHA=$GIT_SHA

RUN apt-get -q update && \
    apt-get autoclean && \
    apt-get -qy --no-install-recommends install build-essential libgc-dev libssl-dev wget curl file jpegoptim optipng gifsicle imagemagick exiftool rsync openssh-client postgresql-client \
    fontconfig ttf-mscorefonts-installer fonts-comic-neue fonts-dejavu fonts-inter fonts-lato fonts-ubuntu && \
    apt-get -y autoremove && \
    apt-get -y clean && \
    fc-cache -fv && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN mkdir -p /app/public
COPY --from=jsbuilder /app/public/*.png /app/public/
COPY --from=jsbuilder /app/public/css /app/public/css
COPY --from=jsbuilder /app/public/js /app/public/js
COPY --from=jsbuilder /app/public/images /app/public/images
COPY --from=jsbuilder /app/public/manifest.json /app/public/manifest.json
RUN chmod -R 755 /app/public

COPY --from=builder /app/bin/worker /app
COPY --from=builder /app/bin/server /app

EXPOSE 3000

ENTRYPOINT [""]