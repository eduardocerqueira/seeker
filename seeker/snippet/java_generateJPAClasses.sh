#date: 2022-05-12T17:13:32Z
#url: https://api.github.com/gists/9d641f54ef5ae3592f6c4f9955b94d23
#owner: https://api.github.com/users/mikolajfranek

rm -R ./target
find ./src/main/java/ -name "*_.java" | xargs rm
mvn compile
cp -R ./target/generated-sources/annotations/* ./src/main/java/