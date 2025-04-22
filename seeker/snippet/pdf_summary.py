#date: 2025-04-22T16:41:25Z
#url: https://api.github.com/gists/01fdcde95458c24bbd2ab4bd17518102
#owner: https://api.github.com/users/vsm-hubble

import boto3
import gradio
from langchain.document_loaders import PyPDFLoader
from langchain.llms.bedrock import Bedrock
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

def get_bedrock_runtime_client():
    return boto3.client(
        service_name="bedrock-runtime",
        region_name="us-west-2",
    )


def summarize_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load_and_split()
    llm = Bedrock(
        model_id="amazon.titan-text-express-v1",
        model_kwargs={
            "maxTokenCount": "**********"
            "stopSequences": [],
            "temperature": 0,
            "topP": 0.9,
        },
        client=get_bedrock_runtime_client(),
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
    )
    chain = load_summarize_chain(llm)
    summary = chain(documents)
    # print(f'{summary} {dict(summary)}')
    return summary["output_text"]


def main():
    input_pdf_path = gradio.File(label="Upload PDF file", type="filepath")
    output_summary = gradio.Textbox(label="Summary")
    gradio.Interface(
        fn=summarize_pdf,
        inputs=input_pdf_path,
        outputs=output_summary,
        title="Summarizer",
        description="This app allows you to summarize your PDF file.",
    ).launch(share=False)


if __name__ == "__main__":
    main()
