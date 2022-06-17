#date: 2022-06-17T17:09:05Z
#url: https://api.github.com/gists/af9399475861bc48cd2164ccaba8b4ce
#owner: https://api.github.com/users/katrbhach

import json
import requests
from app import app
from flask import request
from threading import Thread, Lock
from .config_loader import get_api_interface, get_replicator_connect_url, get_kafka_details, does_env_exist

requests.packages.urllib3.disable_warnings()


@app.route("/replicator/<environment>", methods=["GET"])
def get_replicators(environment):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors", params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch replicators")

        return app.response_class(response=json.dumps({"error": "failed to fetch replicators"}), status=500,
                                  mimetype='application/json')


def does_topic_exist(topic, api_interface: dict, results: dict, result_name: str, thread_lock: Lock):

    results[result_name] = False

    url = api_interface["url"] + "/topics/%s" % topic

    app.logger.info("topic check url is {}".format(url))

    response = requests.get(url, auth=(api_interface["username"], api_interface["password"]), verify=False)

    app.logger.info("topic check response is {}, code is {}".format(response.text, response.status_code))

    if response.status_code == 200:
        thread_lock.acquire()
        results[result_name] = True
        thread_lock.release()


def is_user_permitted_to_access_topic(user: str, topic: str, api_interface: dict, results: dict, result_name: str,
                                      thread_lock: Lock):

    results[result_name] = False

    url = api_interface["url"] + "/acls/search"

    payload = [{
        "topic": topic,
        "user": user,
        "pattern_type": "match"
    }]

    app.logger.info("acl check url is {}".format(url))
    app.logger.info("acl check payload is {}".format(payload))

    response = requests.post(url, json=payload, auth=(api_interface["username"], api_interface["password"]),
                             verify=False)

    app.logger.info("acl check response is {}, code is {}".format(response.text, response.status_code))

    for _ in response.json():
        # Check the ACL here
        thread_lock.acquire()
        results[result_name] = True
        thread_lock.release()


def get_replicator_payload(replicator_name, payload, source_topic, source_kafka_details, destination_kafka_details) -> \
        dict:

    connect_payload = {
        "name": replicator_name,
        "config": {
            "connector.class": "io.confluent.connect.replicator.ReplicatorSourceConnector",
            "tasks.max": payload["overrides"]["tasks.max"] if "tasks.max" in
                                                              payload.get("overrides", dict()) else "1",
            "topic.config.sync": "false",
            "consumer.auto.offset.reset": "earliest",
            "value.converter": "io.confluent.connect.replicator.util.ByteArrayConverter",
            "key.converter": "io.confluent.connect.replicator.util.ByteArrayConverter",
            "topic.whitelist": source_topic,
            "allow.auto.create.topics": "false",
            "topic.auto.create": "false",
            "auto.offset.reset": "earliest",

            "src.kafka.bootstrap.servers": source_kafka_details["bootstrap_servers"],
            "src.kafka.security.protocol": "SASL_PLAINTEXT",
            "src.kafka.sasl.mechanism": "SCRAM-SHA-256",
            "src.kafka.sasl.jaas.config":
                "org.apache.kafka.common.security.scram.ScramLoginModule required username=\"%s\" password=\"%s\";"
                % (source_kafka_details["username"], source_kafka_details["password"]),
            "src.consumer.group.id": replicator_name,
            "src.kafka.client.id": replicator_name,
            "src.consumer.max.poll.records": "2000",
            "src.kafka.reconnect.backoff.ms": "1000",

            "dest.kafka.bootstrap.servers": destination_kafka_details["bootstrap_servers"],
            "producer.override.bootstrap.servers": destination_kafka_details["bootstrap_servers"],
            "dest.kafka.security.protocol": "SASL_PLAINTEXT",
            "producer.override.security.protocol": "SASL_PLAINTEXT",
            "dest.kafka.sasl.mechanism": "SCRAM-SHA-256",
            "producer.override.sasl.mechanism": "SCRAM-SHA-256",
            "producer.override.sasl.jaas.config":
                "org.apache.kafka.common.security.scram.ScramLoginModule required username=\"%s\" password=\"%s\";"
                % (destination_kafka_details["username"], destination_kafka_details["password"]),
            "dest.kafka.sasl.jaas.config":
                "org.apache.kafka.common.security.scram.ScramLoginModule required username=\"%s\" password=\"%s\";"
                % (destination_kafka_details["username"], destination_kafka_details["password"]),
            "dest.kafka.client.id": replicator_name
        }
    }

    if "topic.rename.format" in payload.get("overrides", dict()):
        connect_payload["config"]["topic.rename.format"] = payload["overrides"]["topic.rename.format"]

    if "key.converter" in payload.get("overrides", dict()):
        connect_payload["config"]["key.converter"] = payload["overrides"]["key.converter"]

    if "value.converter" in payload.get("overrides", dict()):
        connect_payload["config"]["value.converter"] = payload["overrides"]["value.converter"]

    return connect_payload


@app.route("/replicator", methods=["POST"])
def create_replicator():

    try:

        payload = request.get_json(force=True)

        pre_checks = list()
        thread_lock = Lock()
        pre_check_results = dict()

        source_env = payload["source_env"]
        destination_env = payload["destination_env"]

        if not does_env_exist(source_env) or not does_env_exist(destination_env):
            return app.response_class(response=json.dumps({
                "error": "Source / Destination environment does not exist in config"
            }), status=400, mimetype='application/json')

        source_topic = payload["topic"]
        if "topic.rename.format" in payload.get("overrides", dict()):
            destination_topic = payload["overrides"]["topic.rename.format"].replace("${topic}", source_topic)
        else:
            destination_topic = source_topic

        source_apis = get_api_interface(source_env)
        destination_apis = get_api_interface(destination_env)

        source_kafka_details = get_kafka_details(source_env)
        destination_kafka_details = get_kafka_details(destination_env)

        pre_checks.append(Thread(target=does_topic_exist,
                                 args=(source_topic, source_apis, pre_check_results,
                                       "Topic existence in source environment", thread_lock)))

        pre_checks.append(Thread(target=does_topic_exist,
                                 args=(destination_topic, destination_apis, pre_check_results,
                                       "Topic existence in destination environment", thread_lock)))

        pre_checks.append(Thread(target=is_user_permitted_to_access_topic,
                                 args=(source_kafka_details["username"], source_topic, source_apis, pre_check_results,
                                       "ACL existence in source environment", thread_lock)))

        pre_checks.append(Thread(target=is_user_permitted_to_access_topic,
                                 args=(destination_kafka_details["username"], destination_topic, destination_apis,
                                       pre_check_results, "ACL existence in destination environment", thread_lock)))

        for _ in pre_checks:
            _.start()

        for _ in pre_checks:
            _.join()

        if any(value is False for value in pre_check_results.values()):
            return app.response_class(response=json.dumps({
                "error": "Following pre-check conditions have failed",
                "failures": [k for k in pre_check_results if pre_check_results[k] is False]
            }), status=400, mimetype='application/json')

        replicator_name = "repl-%s2%s-%s" % (source_env, destination_env, source_topic)

        connect_payload = get_replicator_payload(replicator_name, payload, source_topic, source_kafka_details,
                                                 destination_kafka_details)

        replicator_connect_url = get_replicator_connect_url(source_env)

        app.logger.info("API payload is {}, connect payload used to create replicator is {}".
                        format(json.dumps(payload), json.dumps(connect_payload)))

        response = requests.post(replicator_connect_url + "/connect/connectors", json=connect_payload,
                                 params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype='application/json')

    except Exception:

        app.logger.exception("failed to create replicator with payload {}".format(payload))

        return app.response_class(response=json.dumps({"error": "failed to POST replicator to kafka-connect"}),
                                  status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>", methods=["GET"])
def get_replicator_info(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors/%s" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch info of replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to fetch info of replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/status", methods=["GET"])
def get_replicator_status(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors/%s/status" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch status of replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to fetch status of replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/config", methods=["GET"])
def get_replicator_config(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors/%s/config" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch config of replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to fetch config of replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/restart", methods=["POST"])
def restart_replicator(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.post(replicator_connect_url + "/connect/connectors/%s/restart" % replicator_name,
                                 params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to restart replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to restart replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/pause", methods=["PUT"])
def pause_replicator(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.put(replicator_connect_url + "/connect/connectors/%s/pause" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to pause replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to pause replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/resume", methods=["PUT"])
def resume_replicator(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.put(replicator_connect_url + "/connect/connectors/%s/resume" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to resume replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to resume replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>", methods=["DELETE"])
def delete_replicator(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.delete(replicator_connect_url + "/connect/connectors/%s" % replicator_name,
                                   params=request.args, verify=False)

        if response.status_code < 200 or response.status_code > 299:
            return app.response_class(response=json.dumps({"error": "failed to delete resplicator"}),
                                      status=response.status_code, mimetype="application/json")

        app.logger.info("deletion response is {}, code is {}".format(response.text, response.status_code))

        return app.response_class(response=json.dumps({"message": "success"}), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to delete replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to delete replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/tasks", methods=["GET"])
def get_replicator_tasks(environment: str, replicator_name: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors/%s/tasks" % replicator_name,
                                params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch tasks of replicator {}".format(replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to fetch tasks of replicator"}), status=500,
                                  mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/tasks/<task_id>/status", methods=["GET"])
def get_replicator_task_status(environment: str, replicator_name: str, task_id: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.get(replicator_connect_url + "/connect/connectors/%s/tasks/%s/status" %
                                (replicator_name, task_id), params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to fetch status of task {} of replicator {}".format(task_id, replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to fetch status of replicator-task"}),
                                  status=500, mimetype='application/json')


@app.route("/replicator/<environment>/<replicator_name>/tasks/<task_id>/restart", methods=["POST"])
def restart_replicator_task(environment: str, replicator_name: str, task_id: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.post(replicator_connect_url + "/connect/connectors/%s/tasks/%s/restart" %
                                 (replicator_name, task_id), params=request.args, verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to restart task {} of replicator {}".format(task_id, replicator_name))

        return app.response_class(response=json.dumps({"error": "failed to restart replicator-task"}),
                                  status=500, mimetype='application/json')


@app.route("/replicator/<environment>/restart-all-failed-tasks", methods=["POST"])
def restart_all_failed_tasks_in_cluster(environment: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.post(replicator_connect_url + "/connect/restart-all-failed-tasks", params=request.args,
                                 verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to restart all failed tasks of connect cluster")

        return app.response_class(response=json.dumps({
            "error": "failed to restart all failed tasks of connect cluster"
        }), status=500, mimetype='application/json')


@app.route("/replicator/<environment>/restart-all-failed-replicators", methods=["POST"])
def restart_all_failed_replicators_in_cluster(environment: str):

    try:

        if not does_env_exist(environment):
            return app.response_class(response=json.dumps({"error": "Environment does not exist in config"}),
                                      status=400, mimetype='application/json')

        replicator_connect_url = get_replicator_connect_url(environment)

        response = requests.post(replicator_connect_url + "/connect/restart-all-failed-connectors", params=request.args,
                                 verify=False)

        return app.response_class(response=json.dumps(response.json()), status=response.status_code,
                                  mimetype="application/json")

    except Exception:

        app.logger.exception("failed to restart all failed connectors of connect cluster")

        return app.response_class(response=json.dumps({
            "error": "failed to restart all failed replicators of connect cluster"
        }), status=500, mimetype='application/json')
