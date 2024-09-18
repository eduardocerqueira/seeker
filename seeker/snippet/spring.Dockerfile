#date: 2024-09-18T17:05:04Z
#url: https://api.github.com/gists/124d04536cdeca0cc709c6b43ffd9871
#owner: https://api.github.com/users/Riko07br

# Build maven-----------------------
FROM maven:3.8.4-openjdk-17 AS build

WORKDIR /app

COPY src/main ./src/main

COPY pom.xml ./

RUN mvn clean "-Dmaven.test.skip" package

# openJDK runner--------------------
FROM openjdk:17-jdk-alpine

WORKDIR /app

COPY --from=build /app/target/backend-0.0.1-SNAPSHOT.jar ./app.jar

EXPOSE 8080

ENTRYPOINT ["java","-jar","/app/app.jar"]