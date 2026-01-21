#date: 2026-01-21T17:30:16Z
#url: https://api.github.com/gists/10d50bbc0690bf11fa5ddfb0e6edfe82
#owner: https://api.github.com/users/Yashkarde

text_summarizer.py
from transformers import T5Tokenizer, T5ForConditionalGeneration

def summarize_text(text):
    model_name = "t5-small"
    tokenizer = "**********"
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    input_text = "summarize: " + text

    inputs = "**********"
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=120,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = "**********"=True)
    return summary


if __name__ == "__main__":
    print("--------------TEXT SUMMARY----------------")

    text = """
    National Youth Day, observed every year on January 12th, commemorates the birth anniversary of Swami Vivekananda and highlights the importance of youth in nation-building.
    """

    summary = summarize_text(text)
    print(summary)building.
    """

    summary = summarize_text(text)
    print(summary)