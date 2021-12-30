#date: 2021-12-30T17:07:19Z
#url: https://api.github.com/gists/fe672dd879ce86f5e26e8b0f8371deb1
#owner: https://api.github.com/users/backtrackshubham

wget https://archive.apache.org/dist/kafka/2.3.0/kafka_2.12-2.3.0.tgz


wget https://gist.githubusercontent.com/backtrackshubham/fe672dd879ce86f5e26e8b0f8371deb1/raw/27339d9d9b8d557ffbaa1b6d144003389c1756d9/server.properties
wget https://gist.githubusercontent.com/backtrackshubham/fe672dd879ce86f5e26e8b0f8371deb1/raw/27339d9d9b8d557ffbaa1b6d144003389c1756d9/zookeeper.properties
for i in  {2..9}
do
	KAFKA_DIR="kafka-$((i - 1))"
	mkdir $KAFKA_DIR
	cp kafka_2.12-2.3.0.tgz $KAFKA_DIR
	cd $KAFKA_DIR
	KAFKA_LOGS_DIR="$KAFKA_DIR-logs"
	mkdir -p $KAFKA_LOGS_DIR/kafka-logs $KAFKA_LOGS_DIR/zookeeper
	tar xf kafka_2.12-2.3.0.tgz --strip-components=1
	rm kafka_2.12-2.3.0.tgz
	KAFKA_PORT="909$i"
	ZOOKEEPER_PORT="218$((i - 1))"
	KAFKA_LOGS_DIR="$(pwd)/$KAFKA_LOGS_DIR"
	VALUE_DIR="${KAFKA_LOGS_DIR//\//\\/}"
	sed "s/KAFKA_PORT/$KAFKA_PORT/g" ../server.properties > config/server.properties
	sed "s/ZOOKEEPER_PORT/$ZOOKEEPER_PORT/g" ../server.properties > config/server.properties
	sed "s/KAFKA_LOGS_DIR/$VALUE_DIR/g" ../server.properties > config/server.properties
	sed "s/KAFKA_PORT/$KAFKA_PORT/g" ../zookeeper.properties > config/zookeeper.properties
	sed "s/ZOOKEEPER_PORT/$ZOOKEEPER_PORT/g" ../zookeeper.properties > config/zookeeper.properties
	sed "s/KAFKA_LOGS_DIR/$VALUE_DIR/g" ../zookeeper.properties > config/zookeeper.properties
	cd ..
done
echo "***********************************************Process Complete****************************************************"
ls
echo "***********************************************Process Complete****************************************************"

FINISHING_UP_MESSAGE="Congratulations ! We just setup two independent kafka by the names 

kafka of kafka-1, kafka-2, now you just need to goto each of them and open two terminals for 

each of them, (One for kafka server and another one for zookeeper) it will start two kafka on

localhost:9092, localhost:9093, feeling tired just paste these in to different terminals

./kafka-1/bin/zookeeper-server-start.sh kafka-1/config/zookeeper.properties

./kafka-1/bin/kafka-server-start.sh kafka-1/config/server.properties

./kafka-2/bin/zookeeper-server-start.sh kafka-2/config/zookeeper.properties

./kafka-2/bin/kafka-server-start.sh kafka-2/config/server.properties
"

echo "$FINISHING_UP_MESSAGE"

