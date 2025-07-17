#date: 2025-07-17T17:13:48Z
#url: https://api.github.com/gists/567bb5cf68ae6a2e7e7e627e3b2d5551
#owner: https://api.github.com/users/williamfalinski

import time
import json
import functools
import boto3
from typing import Optional
from botocore.config import Config
from datetime import datetime, timedelta, date
from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
from airflow.providers.amazon.aws.hooks.lambda_function import LambdaHook
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator
from airflow.utils.context import Context


class CustomLambdaFunctionOperator(LambdaInvokeFunctionOperator):
    def __init__(
        self,
        *,
        function_name: str,
        log_type: Optional[str] = None,
        qualifier: Optional[str] = None,
        invocation_type: Optional[str] = None,
        client_context: Optional[str] = None,
        payload: Optional[str] = None,
        aws_conn_id: str = "aws_default",
        execution_timeout: timedelta = timedelta(seconds=60),
        connect_timeout: timedelta = timedelta(seconds=10),
        **kwargs,
    ):
        super().__init__(
            function_name=function_name,
            log_type=log_type,
            qualifier=qualifier,
            invocation_type=invocation_type,
            client_context=client_context,
            payload=payload,
            aws_conn_id=aws_conn_id,
            **kwargs,
        )
        self.execution_timeout = execution_timeout
        self.connect_timeout = connect_timeout

    def log_processor(func):
        @functools.wraps(func)
        def wrapper_decorator(self, *args, **kwargs):
            # Usa AwsBaseHook apenas para pegar a região
            hook = AwsBaseHook(aws_conn_id=self.aws_conn_id, client_type="lambda")
            region_name = hook.get_conn().meta.region_name or "us-east-1"

            # Configura boto3 com timeout e sem retries
            config = Config(
                read_timeout=int(self.execution_timeout.total_seconds()),
                connect_timeout=int(self.connect_timeout.total_seconds()),
                retries={"max_attempts": 0}
            )
            print(config)

            # Usa credentials da EC2 IAM Role (não passa nada manualmente)
            lambda_client = boto3.client("lambda", region_name=region_name, config=config)
            params = {
                "FunctionName": self.function_name,
                "InvocationType": self.invocation_type or "RequestResponse",
                "LogType": self.log_type,
                "Payload": self.payload,
            }

            if self.client_context is not None:
                params["ClientContext"] = self.client_context

            if self.qualifier is not None:
                params["Qualifier"] = self.qualifier

            response = lambda_client.invoke(**params)

            request_id = response["ResponseMetadata"]["RequestId"]
            timeout = self.get_function_timeout()

            print(f"[CustomLambda] Verifying logs for RequestId={request_id}, Timeout={timeout}s")
            self.process_log_events(request_id, timeout)

            # Processa Payload antes de retornar
            payload_stream = response.get("Payload")
            if payload_stream:
                try:
                    payload = payload_stream.read().decode()
                    try:
                        payload = json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                    response["Payload"] = payload
                except Exception as e:
                    print(f"[CustomLambda] Error reading payload: {e}")
                    response["Payload"] = None

            return response

        return wrapper_decorator

    @log_processor
    def execute(self, context: Context):
        # A lógica de invocação está no decorator acima
        pass

    def get_function_timeout(self) -> int:
        hook = AwsBaseHook(aws_conn_id=self.aws_conn_id, client_type="lambda")
        client = hook.get_conn()
        resp = client.get_function_configuration(FunctionName=self.function_name)
        return resp["Timeout"]

    def process_log_events(self, request_id: str, function_timeout: int):
        start_time = int(time.time() * 1000)
        for _ in range(function_timeout):
            response_iterator = self.get_response_iterator(self.function_name, request_id, start_time)
            for page in response_iterator:
                # print(page)
                for event in page.get("events", []):
                    start_time = event["timestamp"]
                    print("[CustomLambda] Log Event:", event["message"])
                    if "REPORT RequestId" in event["message"] and request_id in event["message"]:
                        print("[CustomLambda] Lambda execution finished.")
                        return
            time.sleep(2)

        raise RuntimeError("Lambda REPORT RequestId log not found after timeout.")

    def get_response_iterator(self, function_name: str, request_id: str, start_time: int):
        hook = AwsBaseHook(aws_conn_id=self.aws_conn_id, client_type="logs")
        client = hook.get_conn()
        paginator = client.get_paginator("filter_log_events")
        return paginator.paginate(
            logGroupName=f"/aws/lambda/{function_name}",
            filterPattern=f'"{request_id}"',
            # startTime=start_time + 1,
        )