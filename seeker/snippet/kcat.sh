#date: 2021-11-16T17:06:00Z
#url: https://api.github.com/gists/00fff04b65bd6ef8d81c6805cf1612c4
#owner: https://api.github.com/users/peterkowalski

# Metadata listing mode
kcat -L \
-b $KAFKA_BROKER:9096 \
-X security.protocol=SASL_SSL \
-X sasl.mechanisms=SCRAM-SHA-512 \
-X sasl.username=$USERNAME \
-X sasl.password=$PASSWORD

# Producer mode
kcat -t $KAFKA_TOPIC -P \
-b $KAFKA_BROKER:9096 \
-X security.protocol=SASL_SSL \
-X sasl.mechanisms=SCRAM-SHA-512 \
-X sasl.username=$USERNAME \
-X sasl.password=$PASSWORD

# Consumer mode
kcat -t $KAFKA_TOPIC \
-b $KAFKA_BROKER:9096 \
-f '\nKey (%K bytes): %k\t\nValue (%S bytes): %s\n\Partition: %p\tOffset: %o\n--\n' \
-X security.protocol=SASL_SSL \
-X sasl.mechanisms=SCRAM-SHA-512 \
-X sasl.username=$USERNAME \
-X sasl.password=$PASSWORD