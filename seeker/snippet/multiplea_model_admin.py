#date: 2022-01-04T17:15:23Z
#url: https://api.github.com/gists/264744073fa717ea33d036e03eaf69a9
#owner: https://api.github.com/users/rodrigobertin

def create_model_admin(modeladmin, model, name=None):
    class Meta:
        proxy = True
        app_label = model._meta.app_label

    attrs = {'__module__': '', 'Meta': Meta}

    newmodel = type(name, (model,), attrs)

    admin.site.register(newmodel, modeladmin)
    return modeladmin
