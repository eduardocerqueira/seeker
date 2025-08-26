#date: 2025-08-26T17:10:12Z
#url: https://api.github.com/gists/a72430be17e79603f2128e2a3259ef88
#owner: https://api.github.com/users/mcburton

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(initial_story, mo):
    # "Once upon a time in {place} there lived a boy named {name}. He liked to {verb} and {verb} all day long."
    story_template = mo.ui.text_area(value=initial_story, )

    mo.md(f"""

    # Python Mad Libs - Dr. Vee's kids story.

    Fill out the mad libs below to create a kids story.


    """)

    return (story_template,)


@app.cell
def _(generate_widgets, mo, story_template):
    # Example usage
    #story_template = "Once upon a time in {place} there lived a boy named {name}. He liked to {verb} and {verb} all day long."
    widgets = generate_widgets(story_template.value)

    #runny = mo.ui.run_button()

    # Display the generated widgets
    mo.vstack(widgets.values())
    return (widgets,)


@app.cell
def _(mo):
    runny = mo.ui.run_button(label="Click to generate story.")
    runny
    return (runny,)


@app.cell
def _(initial_story, mo, runny, widgets):
    mo.stop(not runny.value)
    barf = {k: v.value for k,v in widgets.items()}

    mo.md(initial_story.format(**barf))
    return


@app.cell
def _():
    # initial story
    initial_story="""Once upon a time, in a {place1}, there lived a {adjective1} {noun1} who was known for their {adjective2} ability to {verb1}. Every morning, the {noun1} would stroll through the {adjective3} village, greeting everyone with a {adjective4} smile. One sunny day, while {verb2} near the {adjective5} river, a {adjective6} {noun2} appeared, holding a mysterious, shimmering {noun3}. “This {noun3} will {verb3} your deepest {noun4},” it whispered. Intrigued, the {noun1} accepted the gift and suddenly found themselves {verb4} on an unexpected adventure.\n\nTheir journey led them to a {adjective7} forest, where they encountered a {adjective8} {noun5} in distress. The {noun5} explained that a {adjective9} {noun6} had taken their precious {noun7}, and they needed help to {verb5} it back. Bravely, the {noun1} agreed to assist, and together they embarked on a perilous quest.\n\nFinally, after a {adjective10} encounter with the {adjective11} {noun6}, they retrieved the lost {noun7}, discovering the true power of {noun8} and {noun9}. The {noun1} returned home, forever changed by their {adjective12} journey, and the {place1} celebrated their {adjective13} tale for generations.
    """
    return (initial_story,)


@app.cell
def _():
    import marimo as mo
    import re
    return mo, re


@app.cell
def _(mo, re):
    def generate_widgets(story: str):
        # Find all placeholders in the story
        placeholders = re.findall(r'\{(.*?)\}', story)

        return {placeholder: mo.ui.text(value='', label=f"Enter a {placeholder}") for placeholder in placeholders}

    return (generate_widgets,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
