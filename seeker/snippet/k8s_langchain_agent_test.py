#date: 2023-12-19T17:00:16Z
#url: https://api.github.com/gists/6d649279b91e3e0faab7dba72dcebf51
#owner: https://api.github.com/users/thoraxe

import os

# Import things that are needed generically
from langchain.agents import AgentType, initialize_agent
from langchain.tools import tool

import langchain
langchain.debug = True

from dotenv import load_dotenv
load_dotenv()

import kubernetes.client
configuration = kubernetes.client.Configuration()

# Configure API key authorization: "**********"
configuration.api_key['authorization'] = "**********"

# Uncomment below to setup prefix (e.g. Bearer) for API key, if needed
configuration.api_key_prefix['authorization'] = 'Bearer'

# Defining host is optional and default to http://localhost
configuration.host = os.getenv("K8S_CLUSTER_API")

# cluster is using a self-signed cert -- would need to inject CA information
configuration.verify_ssl = False

k8s_client = kubernetes.client.ApiClient(configuration)

# k8s stuff
from src.tools.k8s_explorer.tool import KubernetesOpsModel
from src.tools.k8s_explorer.tool import KubernetesGetAvailableNamespacesTool
from src.tools.k8s_explorer.tool import KubernetesGetObjectNamesTool
from src.query_helpers.yaml_generator import YamlGenerator


k8s_ops_model = KubernetesOpsModel.from_k8s_client(k8s_client=k8s_client)
k8s_ns_tool = KubernetesGetAvailableNamespacesTool(model=k8s_ops_model)
k8s_obj_tool = KubernetesGetObjectNamesTool(model=k8s_ops_model)

@tool
def yaml_generator(query: str) -> str:
    """useful for generating YAML for kuberentes or OpenShift objects"""
    yaml_gen = YamlGenerator()
    return yaml_gen.generate_yaml("1234", query)

tools = [yaml_generator, k8s_ns_tool, k8s_obj_tool]

from utils.model_context import get_watsonx_predictor

#llm = get_watsonx_predictor(
#    model= "**********"=5, verbose=True
#)
#llm = get_watsonx_predictor(
#    model= "**********"=5, verbose=True
#)
#llm = get_watsonx_predictor(
#    model= "**********"=5, verbose=True
#)
#llm = get_watsonx_predictor(
#    model= "**********"=5, verbose=True
#)

from langchain.llms import OpenAI
llm = OpenAI(temperature=0, verbose=True)

agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

#agent.run("Can you give me an OpenShift Limitrange YAML that restricts CPU to 100mi?")
agent.run("What pods are in the openshift-storage namespace?"))

#agent.run("Can you give me an OpenShift Limitrange YAML that restricts CPU to 100mi?")
agent.run("What pods are in the openshift-storage namespace?")