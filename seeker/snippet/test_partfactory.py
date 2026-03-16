#date: 2026-03-16T17:42:17Z
#url: https://api.github.com/gists/cd0f68e1b665833ec1fa02e2742787a0
#owner: https://api.github.com/users/96tm

def test_new_part_create_schema(capsys):
    part = factory.build(dict, FACTORY_CLASS=PartCreateSchemaFactory)
    print((f"{part}"))
    captured = capsys.readouterr()
    assert part is not None
    assert "'name':" in captured.out