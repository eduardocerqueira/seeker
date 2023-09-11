#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

# Define groundedness
grounded = Groundedness(groundedness_provider=openai_gpt35)
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness").on(
    TruLlama.select_source_nodes().node.text # context
).on_output().aggregate(grounded.grounded_statements_aggregator)