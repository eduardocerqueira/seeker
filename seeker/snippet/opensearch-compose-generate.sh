#date: 2024-06-14T16:56:22Z
#url: https://api.github.com/gists/b81385126ba46959701555fe82a141c8
#owner: https://api.github.com/users/sachitsac

#!/bin/bash
# 1. To run this script curl it locally:
#    curl https://gist.githubusercontent.com/dtaivpp/c587d99a2cab441eba0314534ae87c86/raw -o opensearch-compose-bootstrap.sh
# 2. Change it to be executable:
#    chmod +x opensearch-compose-generate.sh
# 3. Run it:
#    ./opensearch-compose-generate.sh 
#
# This will create:
#     - docker-compose.yml file for OpenSearch 
#     - .env file with the OpenSearch password
# 
# 4. Docker compose up to start an OpenSearch instance.
#    docker compose up -d

# Check if env file exists and if not create and fill it.
if [[ ! -e .env ]]; then
    echo "OPENSEARCH_PASSWORD= "**********"
fi

if [[ ! -e docker-compose.yml ]]; then
  echo """
services:
  opensearch:
    image: opensearchproject/opensearch:\${OPENSEARCH_VERSION:-latest}
    container_name: opensearch
    environment:
      discovery.type: single-node
      node.name: opensearch
      OPENSEARCH_JAVA_OPTS: -Xms512m -Xmx512m
      OPENSEARCH_INITIAL_ADMIN_PASSWORD: "**********"
    volumes:
      - opensearch-data:/usr/share/opensearch/data
    ports:
      - 9200:9200
      - 9600:9600
    networks:
      - opensearch-net

  opensearch-dashboards:
    image: opensearchproject/opensearch-dashboards:\${OPENSEARCH_DASHBOARDS_VERSION:-latest}
    container_name: opensearch-dashboards
    ports:
      - 5601:5601
    expose:
      - "5601"
    environment:
      OPENSEARCH_HOSTS: '[\"https://opensearch:9200\"]'
    networks:
      - opensearch-net
    depends_on:
      - opensearch

volumes:
  opensearch-data:

networks:
  opensearch-net:
    driver: bridge

""" >> docker-compose.yml
fi
e

""" >> docker-compose.yml
fi
