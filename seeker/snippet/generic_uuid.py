#date: 2021-10-13T16:53:34Z
#url: https://api.github.com/gists/d2c3d480ea372c6fdd9a494299aebe79
#owner: https://api.github.com/users/dcruzf

class GUID(sa.types.TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses
    BINARY(16), to store UUID.
    """

    cache_ok = True
    impl = sa.types.BINARY

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(
                sa.dialects.postgresql.UUID(as_uuid=True)
            )
        else:
            return dialect.type_descriptor(sa.types.BINARY(16))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == "postgresql":
            return value
        else:
            return value.bytes

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value
        else:
            return uuid.UUID(bytes=value)

    @property
    def python_type(self):
        return uuid.UUID