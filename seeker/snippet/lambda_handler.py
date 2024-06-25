#date: 2024-06-25T16:45:36Z
#url: https://api.github.com/gists/246ba2d8765d61701a9234e07db13f9b
#owner: https://api.github.com/users/flavioti

import json
import boto3

def lambda_handler(event, context):
    """
    Função Lambda para iniciar a execução de um pipeline do SageMaker.
    
    :param event: Evento contendo o nome do pipeline e parâmetros opcionais.
    :param context: Contexto da execução da função Lambda.
    :return: Resposta com o ARN da execução do pipeline ou mensagem de erro.
    """
    # Extrai o nome do pipeline e os parâmetros do evento
    pipeline_name = event.get('pipeline_name')
    pipeline_parameters = event.get('pipeline_parameters', [])

    if not pipeline_name:
        return {
            'statusCode': 400,
            'body': json.dumps('Nome do pipeline não fornecido')
        }

    # Cria um cliente do SageMaker
    sagemaker_client = boto3.client('sagemaker')

    try:
        # Inicia a execução do pipeline
        response = sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=pipeline_parameters
        )

        # Obtém o ID da execução do pipeline
        execution_arn = response['PipelineExecutionArn']
        return {
            'statusCode': 200,
            'body': json.dumps(f'Pipeline Execution ARN: {execution_arn}')
        }

    except boto3.exceptions.Boto3Error as e:
        return {
            'statusCode': 500,
            'body': json.dumps(f'Erro ao iniciar a execução do pipeline: {str(e)}')
        }

# Exemplo de evento para teste local
if __name__ == "__main__":
    event = {
        'pipeline_name': 'meu_pipeline',
        'pipeline_parameters': [
            {'Name': 'param1', 'Value': 'valor1'},
            {'Name': 'param2', 'Value': 'valor2'},
        ]
    }
    context = None
    result = lambda_handler(event, context)
    print(result)
