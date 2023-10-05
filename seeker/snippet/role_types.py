#date: 2023-10-05T17:00:14Z
#url: https://api.github.com/gists/fef7801c46b140e80d60ca079434fe48
#owner: https://api.github.com/users/AlanCoding

import os
import sys

# Django
import django


base_dir = os.path.abspath(  # Convert into absolute path string
    os.path.join(  # Current file's grandparent directory
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir,
    )
)

if base_dir not in sys.path:
    sys.path.insert(1, base_dir)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "awx.settings.development")  # noqa
django.setup()  # noqa

from django.apps import apps
from awx.main.fields import ImplicitRoleField


def print_name(field):
    """For a field which is a role field, give a display name"""
    if not hasattr(f, 'model'):
        return f.name
    return f'{f.model._meta.model_name}.{f.name}'


models = set(apps.get_app_config('main').get_models())

roles = {
    # Note that these are not real fields, so they are created artifically
    'singleton:system_administrator': ImplicitRoleField(name='system_administrator'),
    'singleton:system_auditor': ImplicitRoleField(name='system_administrator')
}

for cls in models:
    for f in cls._meta.get_fields():
        if isinstance(f, ImplicitRoleField):
            roles[print_name(f)] = f

for f in roles.values():
    f.parents = {}
    f.children = {}

for k, f in roles.copy().items():
    if f.parent_role is not None:
        if isinstance(f.parent_role, str):
            parent_roles = [f.parent_role]
        else:
            parent_roles = f.parent_role
        for rel_name in parent_roles:
            if '.' in rel_name:
                components = rel_name.split('.')
                other_field = f.model._meta.get_field(components[0])
                for rel_path in components[1:]:
                    model = other_field.remote_field.model
                    other_field = model._meta.get_field(rel_path)
                other_role_name = print_name(other_field)
            elif ':' in rel_name:
                other_role_name = rel_name
                other_field = roles[other_role_name]
            else:
                other_field = f.model._meta.get_field(rel_name)
                other_role_name = print_name(other_field)

            if not isinstance(other_field, ImplicitRoleField):
                raise Exception(f'field {other_field} is not a role field')
            f.parents[other_role_name] = other_field

    # print(print_name(f))
    # print('  ' + str(f.parents))


for k, f in roles.copy().items():
    for parent_name, parent_field in f.parents.items():
        parent_field.children[print_name(f)] = f


def get_descendents(f):
    ret = f.children.copy()
    for child_name, child_field in f.children.items():
        additional = get_descendents(child_field)
        ret.update(additional)
    return ret


for k, f in roles.copy().items():
    print()
    print(print_name(f))
    for child_name, child_field in get_descendents(f).items():
        print(f'    {child_name}')
