#date: 2022-01-31T17:02:24Z
#url: https://api.github.com/gists/461d0f55665a3e66eb2638accdf225d8
#owner: https://api.github.com/users/Billal-B

class SparkJob(proto.Message):
    main_jar_file_uri = proto.Field(proto.STRING, number=1, oneof="driver",)
    main_class = proto.Field(proto.STRING, number=2, oneof="driver",)
    args = proto.RepeatedField(proto.STRING, number=3,)
    jar_file_uris = proto.RepeatedField(proto.STRING, number=4,)
    file_uris = proto.RepeatedField(proto.STRING, number=5,)
    archive_uris = proto.RepeatedField(proto.STRING, number=6,)
    properties = proto.MapField(proto.STRING, proto.STRING, number=7,)
    logging_config = proto.Field(proto.MESSAGE, number=8, message="LoggingConfig",)
