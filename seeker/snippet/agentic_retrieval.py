#date: 2025-07-02T17:10:44Z
#url: https://api.github.com/gists/55a5dba3dd6eda172285afe7fa44b458
#owner: https://api.github.com/users/esemsc-dh324

from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.agent import KnowledgeAgentRetrievalClient
from azure.search.documents.agent.models import (
    KnowledgeAgentRetrievalRequest,
    KnowledgeAgentMessage,
    KnowledgeAgentMessageTextContent,
    KnowledgeAgentIndexParams,
)
from azure.search.documents.indexes.models import (
    KnowledgeAgent,
    KnowledgeAgentAzureOpenAIModel,
    KnowledgeAgentTargetIndex,
    AzureOpenAIVectorizerParameters,
)

from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from synthetic_data.utils.get_env_var import get_env_var
from synthetic_data.utils.logger_config import logger



def create_knowledge_agent(agent_name: str, course_id: str) -> None:
    logger.debug(f"Creating knowledge agent named {agent_name}...")
    search_index_client = SearchIndexClient(
        endpoint=get_env_var("AZURE_SEARCH_ENDPOINT"),
        credential=AzureKeyCredential(get_env_var("AZURE_SEARCH_KEY")),
    )
    agent = KnowledgeAgent(
        name=agent_name,
        models=[
            KnowledgeAgentAzureOpenAIModel(
                azure_open_ai_parameters=AzureOpenAIVectorizerParameters(
                    resource_url=get_env_var("AZURE_OPENAI_PORTAL_ENDPOINT"),
                    deployment_name="gpt-4.1-mini",
                    model_name="gpt-4.1-mini",
                )
            )
        ],
        target_indexes=[
            KnowledgeAgentTargetIndex(
                index_name=course_id.lower(), default_reranker_threshold=2.5
            )
        ],
    )

    search_index_client.create_or_update_agent(agent)
    logger.info(f"Knowledge agent {agent_name} created successfully.")


def main():
    agent_name = "knowledge-agent"
    course_id = "MAT901"
    create_knowledge_agent(
        agent_name=agent_name,
        course_id=course_id,
    )

    logger.debug("Creating Knowledge Agent Retrieval Client...")
    agent_client = KnowledgeAgentRetrievalClient(
        endpoint=get_env_var("AZURE_SEARCH_ENDPOINT"),
        agent_name=agent_name,
        credential=AzureKeyCredential(get_env_var("AZURE_SEARCH_KEY")),
    )
    logger.info("Knowledge Agent Retrieval Client created successfully.")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that interacts with a student, with the goal of making them understand the course syllabus so they succeed in their exam."
        },
        {
            "role": "user",
            "content": "Can you give me the dice roll example that has been given in the course syllabus, and what is it used to understand?",
        }
    ]

    logger.debug("Retrieving information from the knowledge agent...")
    retrieval_result = agent_client.retrieve(
        retrieval_request=KnowledgeAgentRetrievalRequest(
            messages=[
                KnowledgeAgentMessage(
                    role=msg["role"],
                    content=[
                        KnowledgeAgentMessageTextContent(text=msg["content"])
                    ],
                )
                for msg in messages
                if msg["role"] != "system"
            ],
            target_index_params=[
                KnowledgeAgentIndexParams(
                    index_name=course_id.lower(), reranker_threshold=2.5
                )
            ],
        )
    )
    messages.append(
        {
            "role": "assistant",
            "content": retrieval_result.response[0].content[0].text, # type: ignore
        }
    )

    print("Response")
    print(retrieval_result.response[0].content[0].text) # type: ignore


if __name__ == "__main__":
    main()
