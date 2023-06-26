#date: 2023-06-26T17:02:52Z
#url: https://api.github.com/gists/51b94c15fdd89c27622118e32a189180
#owner: https://api.github.com/users/quetwo

FROM lucee/lucee:5.4.0.80-light
LABEL authors="nk"

ENV LUCEE_EXTENSIONS "D46B46A9-A0E3-44E1-D972A04AC3A8DC10;name=Chart;version=1.0.19.24,B737ABC4-D43F-4D91-8E8E973E37C40D1B;name=Images;version=2.0.0.25,FAD1E8CB-4F45-4184-86359145767C29DE;name=ORM;version=3.5.5.89,A03F4335-BDEF-44DE-946FB16C47802F96;name=JSON;version=1.0.0.0-RC,7E673D15-D87C-41A6-8B5F1956528C605F;name=MySQL;version=8.0.33"

RUN rm -R /var/www
RUN ln -sf /dev/stdout /opt/lucee/web/logs/application.log && ln -sf /dev/stdout /opt/lucee/web/logs/exception.log
RUN ln -sf /dev/stdout /opt/lucee/server/lucee-server/context/logs/application.log && ln -sf /dev/stdout /opt/lucee/server/lucee-server/context/logs/out.log
RUN echo "echo \$LUCEE_ADMIN_PASSWORD > /opt/lucee/server/lucee-server/context/password.txt" >> /usr/local/tomcat/bin/setenv.sh
COPY ./webroot/ /var/www/

RUN /usr/local/tomcat/bin/prewarm.sh

EXPOSE 8888