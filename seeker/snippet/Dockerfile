#date: 2021-11-04T17:05:06Z
#url: https://api.github.com/gists/606a7c048afc7be557475d3add4e2e6f
#owner: https://api.github.com/users/furkanh

FROM maven:3-jdk-11 AS build
WORKDIR /app
COPY ./ ./
RUN mvn clean package

FROM openjdk:11
WORKDIR /app
COPY --from=build /app/target/service.jar ./
CMD ["java","-jar","service.jar"]