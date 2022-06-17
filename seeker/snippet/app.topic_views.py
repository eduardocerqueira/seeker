#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

import json
import requests
import time

from app import app, mrc_config
from flask import request
from threading import Thread, Lock
from .env_constants import EnvironmentConstants
from .kafka_logger import audit_logger


requests.packages.urllib3.disable_warnings()
mrc_pointer = 0


@app.route("/topics", methods=["GET"])
def get_all_topics():

    try:

        start_time = time.time()

        response = requests.get("{}/topics".format(EnvironmentConstants.url),
                                auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                verify=False, timeout=30)

        app.logger.info("completed fetching topics in {} s".format(time.time() - start_time))

        status_code = response.status_code
        response = response.json()

        if 200 <= status_code <= 299:
            response = [i["topic_name"] for i in response['data']]

        return app.response_class(response=json.dumps(response), status=status_code, mimetype='application/json')

    except Exception:

        app.logger.exception("failed to fetch topics")

        return app.response_class(response=json.dumps({"error": "failed to fetch topics"}),
                                  status=500,
                                  mimetype='application/json')


def get_topic_configs(topic: str, result_dict: dict, error_dict: dict, thread_lock: Lock):

    try:

        start_time = time.time()

        response = requests.get("{}/topics/{}/configs".format(EnvironmentConstants.url, topic),
                                auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                verify=False, timeout=30)

        thread_lock.acquire()

        status_code = response.status_code

        if status_code > 299 or status_code < 200:
            error_dict["configs"] = {"code": status_code, "resp": response.json()}
        else:
            app.logger.info("completed fetching config of topic: {} in {} s".format(topic, time.time() - start_time))
            result_dict['configs'] = response.json()

        thread_lock.release()

    except Exception:

        app.logger.exception("failed to fetch config of topic: {}".format(topic))


def get_topic_partitions(topic: str, result_dict: dict, error_dict: dict, thread_lock: Lock):

    try:

        start_time = time.time()

        response = requests.get("{}/topics/{}/partitions".format(EnvironmentConstants.url, topic),
                                auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                verify=False, timeout=30)

        thread_lock.acquire()

        status_code = response.status_code

        if status_code > 299 or status_code < 200:
            error_dict["partitions"] = {"code": status_code, "resp": response.json()}
        else:
            app.logger.info("completed fetching partitions of topic: {} in {} s".
                            format(topic, time.time() - start_time))
            result_dict['partitions'] = response.json()

        thread_lock.release()

    except Exception:

        app.logger.exception("failed to fetch partitions of topic: {}".format(topic))


@app.route("/topics/<topic>", methods=["GET"])
def get_topic_details(topic: str):

    try:

        start_time = time.time()

        thread_lock = Lock()

        combined_result = dict()
        combined_errors = dict()

        apis_to_call = [
            Thread(target=get_topic_configs, args=(topic, combined_result, combined_errors, thread_lock)),
            Thread(target=get_topic_partitions, args=(topic, combined_result, combined_errors, thread_lock))
        ]

        # Starting all API requests in their own threads
        for api in apis_to_call:
            api.start()

        # waiting for threads to complete their job
        for api in apis_to_call:
            api.join()

        if combined_errors:
            for _, value in combined_errors:
                return app.response_class(response=json.dumps(value["resp"]), status=value["code"],
                                          mimetype='application/json')

        app.logger.info("completed fetching all the details of topic: {} in {} s".
                        format(topic, time.time() - start_time))

        if "configs" not in combined_result or "partitions" not in combined_result:
            raise Exception("Failed to fetch partitions/configs")

        final_result = dict()

        final_result["topicName"] = topic
        final_result["numPartitions"] = str(len(combined_result["partitions"]["data"]))

        for config in combined_result["configs"]["data"]:
            if config['name'] == "cleanup.policy":
                final_result["cleanupPolicy"] = config["value"]
            elif config["name"] == "retention.ms":
                final_result["retentionMS"] = config["value"]
            elif config["name"] == "retention.bytes":
                final_result["retentionBytes"] = config["value"]
            elif mrc_config["mrc_enabled"] and config["name"] == "confluent.placement.constraints" and \
                    config["value"] != "":
                final_result["confluent.placement.constraints"] = json.loads(config["value"])

        return app.response_class(response=json.dumps(final_result), status=200, mimetype='application/json')

    except Exception:

        app.logger.exception("failed to fetch details of topic: {}".format(topic))

        return app.response_class(response=json.dumps({"error": "failed to fetch topic details from kafka-rest-proxy"}),
                                  status=500,
                                  mimetype='application/json')


@app.route("/topics/<topic>", methods=["DELETE"])
def delete_topic(topic: str):

    try:

        start_time = time.time()

        payload = request.get_json(force=True)

        _ = payload["requestedFor"]
        _ = payload["requestNumber"]

        response = requests.delete("{}/topics/{}".format(EnvironmentConstants.url, topic),
                                   auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                   verify=False, timeout=30)

        status_code = response.status_code
        if status_code > 299 or status_code < 200:
            return app.response_class(response=json.dumps(response.json()), status=status_code,
                                      mimetype='application/json')

        app.logger.info("completed deleting topic: {} in {} s for env: {}, owner: {} based on request-number: {}".
                        format(topic, time.time() - start_time, EnvironmentConstants.env_name,
                        payload.get("requestedFor", ""), payload.get("requestNumber", "")))

        audit_logger.info({
            "source": "kafka-topics-apis",
            "op_entity_type": "topic",
            "op": "delete",
            "op_entity_name": topic,
            "env": EnvironmentConstants.env_name,
            "payload": payload,
            "send_to_mongo": True
        })

        start_time = time.time()

        acl_delete_payload = [{
            "topic": topic,
            "pattern_type": "LITERAL"
        }]

        response = requests.delete("{}/acls-internal".format(EnvironmentConstants.acl_api_url), json=acl_delete_payload,
                                   verify=False, timeout=30)

        if response.status_code > 299 or response.status_code < 200:
            app.logger.error("failed to delete all ACLs of topic: {}".format(topic))
        else:
            app.logger.info("completed deleting all ACLs of topic: {} in {} s for env: {}".
                            format(topic, time.time() - start_time, EnvironmentConstants.env_name))

        return app.response_class(response=json.dumps({"message": "success"}), status=status_code,
                                  mimetype='application/json')

    except Exception:

        app.logger.exception("failed to delete topic: {}".format(topic))

        return app.response_class(response=json.dumps({"error": "failed to delete topic from kafka-rest-proxy"}),
                                  status=500,
                                  mimetype='application/json')


@app.route("/topics", methods=["POST"])
def create_topic():

    try:

        start_time = time.time()

        payload = request.get_json(force=True)

        _ = payload["requestedFor"]
        _ = payload["requestNumber"]

        kafka_rest_proxy_payload = {
            "topic_name": payload["topicName"].lower() if payload.get("caseOverride", False) else payload["topicName"],
            "partitions_count": int(payload["numPartitions"]),
            "configs": [
                {
                    "name": "cleanup.policy",
                    "value": payload["cleanupPolicy"]
                },
                {
                    "name": "retention.ms",
                    "value": payload["retentionMS"]
                },
                {
                    "name": "retention.bytes",
                    "value": payload.get("retentionBytes", "-1")
                }
            ]
        }

        if mrc_config["mrc_enabled"]:
            global mrc_pointer
            kafka_rest_proxy_payload["configs"].append({
                "name": "confluent.placement.constraints",
                "value": json.dumps(mrc_config["mrc_configs"][mrc_pointer])
            })
            mrc_pointer += 1
            if mrc_pointer >= len(mrc_config["mrc_configs"]):
                mrc_pointer = 0

        app.logger.debug("Payload being sent to rest-proxy to create topic: {}".format(kafka_rest_proxy_payload))

        response = requests.post("{}/topics".format(EnvironmentConstants.url),
                                 auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                 verify=False, timeout=30,
                                 json=kafka_rest_proxy_payload)

        status_code = response.status_code
        if status_code > 299 or status_code < 200:
            return app.response_class(response=json.dumps(response.json()), status=status_code,
                                      mimetype='application/json')

        app.logger.info("completed creating topic: {} in {} s for env: {}, owner: {} based on request-number: {}".
                        format(payload["topicName"], time.time() - start_time, EnvironmentConstants.env_name,
                               payload.get("requestedFor", ""), payload.get("requestNumber", "")))

        audit_logger.info({
            "source": "kafka-topics-apis",
            "op_entity_type": "topic",
            "op": "create",
            "op_entity_name": payload["topicName"],
            "env": EnvironmentConstants.env_name,
            "payload": payload,
            "send_to_mongo": True
        })

        return app.response_class(response=json.dumps({"message": "success"}), status=status_code,
                                  mimetype='application/json')

    except Exception:

        app.logger.exception("failed to create topic with payload {}".format(payload))

        return app.response_class(response=json.dumps({"error": "failed to POST topic to kafka-rest-proxy"}),
                                  status=500,
                                  mimetype='application/json')


@app.route("/topics/<topic>/configs", methods=["PUT"])
def alter_topic_configs(topic):

    try:

        start_time = time.time()

        payload = request.get_json(force=True)

        _ = payload["requestedFor"]
        _ = payload["requestNumber"]

        kafka_rest_proxy_payload = {
            "data": [
                {
                    "name": "cleanup.policy",
                    "value": payload["cleanupPolicy"]
                },
                {
                    "name": "retention.ms",
                    "value": payload["retentionMS"]
                },
                {
                    "name": "retention.bytes",
                    "value": payload.get("retentionBytes", "-1")
                }
            ]
        }

        app.logger.debug("Payload being sent to rest-proxy to alter topic: {}".format(kafka_rest_proxy_payload))

        response = requests.post("{}/topics/{}/configs:alter".format(EnvironmentConstants.url, topic),
                                 auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                 verify=False, timeout=30,
                                 json=kafka_rest_proxy_payload)

        status_code = response.status_code
        if status_code > 299 or status_code < 200:
            return app.response_class(response=json.dumps(response.json()), status=status_code,
                                      mimetype='application/json')

        app.logger.info("completed altering config of topic: {} in {} s for env: {}, owner: {} "
                        "based on request-number: {}".format(topic, time.time() - start_time,
                                                             EnvironmentConstants.env_name,
                                                             payload.get("requestedFor", ""),
                                                             payload.get("requestNumber", "")))

        audit_logger.info({
            "source": "kafka-topics-apis",
            "op_entity_type": "topic",
            "op": "alter",
            "op_entity_name": topic,
            "env": EnvironmentConstants.env_name,
            "payload": payload,
            "send_to_mongo": True
        })

        return app.response_class(response=json.dumps({"message": "success"}), status=status_code,
                                  mimetype='application/json')

    except Exception:

        app.logger.exception("failed to alter config of topic: {}".format(topic))

        return app.response_class(response=json.dumps({"error": "failed to alter topic config"}),
                                  status=500,
                                  mimetype='application/json')
