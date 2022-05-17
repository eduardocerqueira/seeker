#date: 2022-05-17T16:52:09Z
#url: https://api.github.com/gists/7d51255f800a47801ba2744e7d6d73a2
#owner: https://api.github.com/users/gustavo-flor

FROM maven:3.8.5-openjdk-17-slim AS build

COPY pom.xml .
COPY src src

RUN mvn clean install --batch-mode -DskipTests

FROM openjdk:17-slim AS release

COPY --from=build /target/*.jar /app.jar
COPY docker-entrypoint.sh /

RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]