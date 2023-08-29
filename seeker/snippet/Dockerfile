#date: 2023-08-29T16:51:36Z
#url: https://api.github.com/gists/401abd85acd2ec2a02fcc529f23b5f03
#owner: https://api.github.com/users/matiasmasca

FROM dev as test

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
  wget \
  unzip \
  apt-utils \
  libasound2 \
  libatk1.0-0 \
  libc6 \
  libcairo2 \
  libcups2 \
  libdbus-1-3 \
  libexpat1 \
  libfontconfig1 \
  libgcc1 \
  libgconf-2-4 \
  libgdk-pixbuf2.0-0 \
  libglib2.0-0 \
  libgtk-3-0 \
  libnspr4 \
  libpango-1.0-0 \
  libpangocairo-1.0-0 \
  libstdc++6 \
  libx11-6 \
  libx11-xcb1 \
  libxcb1 \
  libxcursor1 \
  libxdamage1 \
  libxext6 \
  libxfixes3 \
  libxi6 \
  libxrandr2 \
  libxrender1 \
  libxss1 \
  libxtst6 \
  libnss3 \
  iproute2 \
  net-tools \
  unzip xvfb libxi6 libgconf-2-4 jq libjq1 libonig5 libxkbcommon0 libxss1 libglib2.0-0 libnss3 \
  libfontconfig1 libatk-bridge2.0-0 libatspi2.0-0 libgtk-3-0 libpango-1.0-0 libgdk-pixbuf2.0-0 libxcomposite1 \
  libxcursor1 libxdamage1 libxtst6 libappindicator3-1 libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libxfixes3 \
  libdbus-1-3 libexpat1 libgcc1 libnspr4 libgbm1 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 libxext6 \
  libxrandr2 libxrender1 gconf-service ca-certificates fonts-liberation libappindicator1 lsb-release xdg-utils \
  # The following are used to trim down the size of the image by removing unneeded data
  && apt-get clean autoclean \
  && apt-get autoremove -y \
  && rm -rf \
    /var/lib/apt \
    /var/lib/dpkg \
    /var/lib/cache \
    /var/lib/log

RUN bundle install --jobs 20 --retry 5

# #Chrome & Chrome Driver
# 114 
# https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/114.0.5735.90/linux64/chrome-linux64.zip
# https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
# 115 
# https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/115.0.5790.170/linux64/chrome-linux64.zip
# https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/115.0.5790.170/linux64/chromedriver-linux64.zip 

# Install Chrome
RUN wget -O /tmp/google-chrome.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/115.0.5790.170/linux64/chrome-linux64.zip && \
    unzip /tmp/google-chrome.zip -d /opt/ && \
    rm /tmp/google-chrome.zip
RUN chmod +x /opt/chrome-linux64/chrome
RUN ln -s /opt/chrome-linux64/chrome /usr/bin/google-chrome

# Put Chrome into the PATH
ENV PATH="/opt/chrome-linux64:${PATH}"

# Install Chromedriver
RUN wget -O /tmp/chromedriver-linux64.zip https://edgedl.me.gvt1.com/edgedl/chrome/chrome-for-testing/115.0.5790.170/linux64/chromedriver-linux64.zip && \
   unzip /tmp/chromedriver-linux64.zip -d /tmp/chromedriver-linux64/ && \
   mv /tmp/chromedriver-linux64/chromedriver-linux64/chromedriver /usr/bin/ && \
   chmod +x /usr/bin/chromedriver
   
# Put Chromedriver into the PATH
ENV PATH /usr/bin/:$PATH

COPY ./script/wrap-chrome-binary /opt/bin/wrap-chrome-binary
RUN chmod +x /opt/bin/wrap-chrome-binary
RUN /opt/bin/wrap-chrome-binary

COPY . ./