#date: 2024-07-02T16:36:15Z
#url: https://api.github.com/gists/3d02b0cd8c372809e84aa377f4029ffc
#owner: https://api.github.com/users/cameronboy

from dspy.teleprompt import MIPRO
import dspy
import re
import logging

logging.basicConfig(
    filename="logging.log",
    level=logging.DEBUG,
    format="%(asctime)s; %(levelname)s; %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


class Categorize(dspy.Signature):
    """Classify the provided feedback into one or more labels"""
    comment = dspy.InputField(desc="user provided feedback to the question: 'what can we do better?'")
    labels = dspy.OutputField(desc="String of '|' separated categories the feedback falls into")


class CategorizeFeedback(dspy.Module):
    """Classify the provided feedback into one or more labels"""

    def __init__(self):
        super().__init__()
        self.generate_labels = dspy.Predict(Categorize)

    def forward(self, comment):
        prediction = self.generate_labels(comment=comment)
        return dspy.Prediction(answer=prediction.labels)


def f1_score_metric(gold, pred, trace=None):
    logging.info("F1 Score Args")
    logging.info(f"gold: {gold}")
    logging.info(f"pred: {pred}")

    # ...Rest of function
    return f1


NUM_THREADS = 1
model_to_generate_prompts = lm
model_that_solves_task = lm
your_defined_metric = f1_score_metric
prompt_generation_temperature = .5
num_new_prompts_generated = 1

teleprompter = MIPRO(
    prompt_model=model_to_generate_prompts,
    task_model=model_that_solves_task,
    metric=your_defined_metric,
    num_candidates=num_new_prompts_generated,
    init_temperature=prompt_generation_temperature)

kwargs = dict(
    num_threads=NUM_THREADS,
    display_progress=True,
    display_table=0)


compiled_categorizer = teleprompter.compile(
    CategorizeFeedback(),
    trainset=trainset,
    num_trials=100, max_bootstrapped_demos=3,
    max_labeled_demos=5, eval_kwargs=kwargs)