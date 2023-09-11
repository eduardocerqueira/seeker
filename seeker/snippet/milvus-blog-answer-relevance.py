#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name = "Answer Relevance").on_input_output()