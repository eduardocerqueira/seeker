#date: 2025-09-16T16:58:59Z
#url: https://api.github.com/gists/02a62544692e57d278c6b63bf87fb42f
#owner: https://api.github.com/users/AlexandrDragunkin

from __future__ import annotations

import builtins
from collections import deque
from dataclasses import field as dataclass_field, Field, MISSING
from logging import getLogger


log = getLogger(__name__)


def field_property_support(name, bases=None, cls_dict=None) -> type:
    """
    A metaclass which ensures that dataclass fields which are associated with
    properties are properly supported as expected.
    Accepts the same arguments as the builtin `type` function::
        type(name, bases, dict) -> a new type
    Check out my answer for more background info:
      - https://stackoverflow.com/a/69847295/10237506
    """

    # annotations can also be forward-declared, i.e. as a string
    cls_annotations: dict[str, type | str] = cls_dict['__annotations__']
    # we're going to be doing a lot of `append`s, so might be better to use a
    # deque here rather than a list.
    body_lines: deque[str] = deque()
    # keeps track of whether we've seen a field with a default value already.
    # since field properties ideally shouldn't come after default fields (as
    # the property definition overwrites any default value), we should raise
    # a helpful error in such cases.
    seen_default_value = False

    # does the class define a __post_init__() ?
    if '__post_init__' in cls_dict:
        has_post_init = True
        fn_locals = {'_orig_post_init': cls_dict['__post_init__']}
    else:
        # else, we set a 'do-nothing' __post_init__() after an initial run.
        def __post_init__(*_):
            pass

        has_post_init = False
        fn_locals = {'_orig_post_init': __post_init__}

    # Loop over and identify all dataclass fields with associated properties.
    for field, annotation in cls_annotations.items():
        if field in cls_dict:
            fval = cls_dict[field]
            if isinstance(fval, property):
                # Add property object to __post_init__() locals
                fn_locals[f'property_{field}'] = fval
                # 1. save the dataclass field value to a local var
                body_lines.append(f'{field} = self.{field}')
                # 2. set the property (for the field) on the class
                body_lines.append(f'cls.{field} = property_{field}')
                # 3. call the property setter with the field value
                body_lines.append(f'self.{field} = {field}')
                # check if the field with a leading underscore is assigned a
                # value; if so, we can use this to assign a default value to
                # the field property.
                under_f = '_' + field
                if under_f in cls_dict:
                    uf_val = cls_dict[under_f]
                    if isinstance(uf_val, Field):
                        if uf_val.default is not MISSING:
                            cls_dict[field] = dataclass_field(default=uf_val.default)
                        elif uf_val.default_factory is not MISSING:
                            cls_dict[field] = dataclass_field(
                                default_factory=uf_val.default_factory
                            )
                        else:
                            del cls_dict[field]
                    else:  # underscored field value is not a dataclass.Field
                        cls_dict[field] = uf_val
                elif seen_default_value:
                    # non-default field property follows a default field
                    ann_name = getattr(annotation, '__qualname__', annotation)
                    msg = (
                        f"non-default argument '{field}' follows default argument.\n\n"
                        f"resolution: define a default value for the field property as shown below.\n"
                        f"  {field}: {ann_name!s}\n"
                        f"  _{field} = None"
                    )
                    raise TypeError(msg)
                else:
                    # else, no default value is specified for the field, which
                    # means it's a *required* parameter in __init__(). we
                    # should now clear the property object, so `dataclasses`
                    # doesn't add it as a default value in __init__().
                    del cls_dict[field]
            else:
                seen_default_value = True

    # only add a __post_init__() if there are field properties in the class
    if not body_lines:
        cls = type(name, bases, cls_dict)
        return cls

    body_lines.appendleft('cls = self.__class__')
    # on an initial run, call the class's original __post_init__()
    if has_post_init:
        body_lines.append('_orig_post_init(self, *args)')

    # Our generated __post_init__() only runs the first time that __init__()
    # runs. Now we can either set a 'do-nothing' __post_init__(), or set the
    # original __post_init__() method that was defined in the class.
    body_lines.append('cls.__post_init__ = _orig_post_init')

    # generate a new __post_init__() method
    _post_init_fn = _create_fn(
        '__post_init__',
        # ensure that we accept variadic positional arguments, in case
        # the class already defines a __post_init__()
        ('self', '*args',),
        body_lines,
        globals=cls_dict,
        locals=fn_locals,
        return_type=None,
    )

    # Set the __post_init__() attribute on the class
    cls_dict['__post_init__'] = _post_init_fn

    # (Optional) Print the body of the generated method definition
    log.debug('Generated a body definition for %s.__post_init__():', name)
    log.debug('%s\n  %s', '-------', '\n  '.join(body_lines))
    log.debug('-------')

    cls = type(name, bases, cls_dict)
    return cls


# Note: this helper function is copied verbatim. The original implementation
# is from the `dataclasses` module in Python 3.9.
def _create_fn(name, args, body, *, globals=None, locals=None,
               return_type=MISSING):
    # Note that we mutate locals when exec() is called.  Caller
    # beware!  The only callers are internal to this module, so no
    # worries about external callers.
    if locals is None:
        locals = {}
    if 'BUILTINS' not in locals:
        locals['BUILTINS'] = builtins
    return_annotation = ''
    if return_type is not MISSING:
        locals['_return_type'] = return_type
        return_annotation = '->_return_type'
    args = ','.join(args)
    body = '\n'.join(f'  {b}' for b in body)

    # Compute the text of the entire function.
    txt = f' def {name}({args}){return_annotation}:\n{body}'

    local_vars = ', '.join(locals.keys())
    txt = f"def __create_fn__({local_vars}):\n{txt}\n return {name}"

    ns = {}
    exec(txt, globals, ns)
    return ns['__create_fn__'](**locals)
