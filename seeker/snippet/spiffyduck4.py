#date: 2022-01-03T17:01:35Z
#url: https://api.github.com/gists/c8e1eda877772aa3ebdda254e07a6b49
#owner: https://api.github.com/users/danfunk

def show_form(task):
    model = {}
    form = task.task_spec.form

    if task.data is None:
        task.data = {}

    for field in form.fields:
        prompt = field.label
        if isinstance(field, EnumFormField):
            prompt += "? (Options: " + ', '.join([str(option.id) for option in field.options]) + ")"
        prompt += "? "
        answer = input(prompt)
        if field.type == "long":
            answer = int(answer)
        task.update_data_var(field.id,answer)
