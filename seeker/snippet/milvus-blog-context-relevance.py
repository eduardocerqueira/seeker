#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

# Question/statement relevance between question and each context chunk.
f_context_relevance = Feedback(openai_gpt35.qs_relevance_with_cot_reasons, name = "Context Relevance").on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.max)