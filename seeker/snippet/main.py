#date: 2026-02-03T17:42:00Z
#url: https://api.github.com/gists/c72522ce3f0453a6ad6725c569bd364e
#owner: https://api.github.com/users/datavudeja

# coding: utf-8
import re
from sqlalchemy import cast, literal
from sqlalchemy.dialects.postgresql import array
from sqlalchemy.types import SchemaType, TypeDecorator, Enum
from sqlalchemy.util import set_creation_order, OrderedDict
from psycopg2._psycopg import new_type, new_array_type, register_type


class DeclarativeEnum(object):
    "Declarative enumeration."

    __metaclass__ = DeclEnumMeta
    _reg = OrderedDict()

    @classmethod
    def db_type(cls):
        return DeclarativeEnumType(cls)

    @classmethod
    def from_string(cls, value):
        try:
            return cls._reg[value]
        except KeyError:
            raise ValueError("Invalid value for {!r}: {!r}".format(cls.__name__, value))

    @classmethod
    def names(cls):
        return cls._reg.keys()

    @classmethod
    def choices(cls):
        return cls._reg.items()

    # REF: psycopg2/_json.py
    @classmethod
    def register(cls, db):
        typname = str(cls.db_type().impl.name)

        try:
            result = db.session.execute("SELECT t.oid, typarray FROM pg_type t WHERE t.typname = '{}';".format(typname))
            oid, array_oid = result.cursor.fetchone()

            _type = new_type((oid, ), typname, lambda s, cur: s)
            register_type(_type)

            if array_oid:
                result = db.session.execute("SELECT typname FROM pg_type t WHERE t.oid = '{}';".format(array_oid))
                array_typname = str(result.cursor.fetchone()[0])

                _type_array = new_array_type((array_oid,), array_typname, _type)
                register_type(_type_array)

            db.session.rollback()
        except TypeError:
            pass


class DeclarativeEnumArrayType(TypeDecorator):
    enumTypeCls = None
    impl = None

    def bind_expression(self, bind_value):
        val = bind_value.effective_value
        if val is None:
            val = []
        elif not hasattr(val, '__iter__'):
            return cast(bind_value, self.__class__)
        return array(cast(literal(str(ele)), self.__class__.enumTypeCls.db_type()) for ele in val)
