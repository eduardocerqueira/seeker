#date: 2021-11-30T17:02:14Z
#url: https://api.github.com/gists/9fd6c8619fd6b4c8f43c7737c809d237
#owner: https://api.github.com/users/gitmadhu

FROM openjdk:8-jdk-alpine

ARG DEPENDENCY=target
COPY ${DEPENDENCY}/mtodo-1.0-SNAPSHOT.jar /app.jar
ENTRYPOINT ["java","-jar","-Dspring.profiles.active=dev","/app.jar"]