#date: 2023-03-22T17:01:11Z
#url: https://api.github.com/gists/31a8df58afdcab4939972fddc770c5ee
#owner: https://api.github.com/users/ajhebert

from os import environ
from contextvars import copy_context
from context import hello, my_var, MySettings


def test_context_setting():
    """
    Determine if Field.default_factory is called within context.
    """
    try:
        # show my_var has no value set yet
        _ = my_var.get()
    except LookupError:
        pass
    finally:
        my_var.set(hello("Context"))

    # check default value of my_var, set in context.py
    assert my_var.get() == (
        expect := hello("Context")
    ), f"expected {expect!r}, saw {my_var.get()!r}"

    # verify default used
    settings = MySettings()
    assert (
        settings.var == my_var.get()
    ), f"expected match, eval'd `{settings.var!r} == {my_var.get()!r}`"

    # setting a ContextVar() within a Context() affects subsequent runs
    test_ctx = copy_context()
    test_ctx.run(my_var.set, hello("Pytest"))

    # check we have not affected my_var in its original context
    assert my_var.get() == (
        expect := hello("Context")
    ), f"expected {expect!r}, saw {my_var.get()!r}"

    @test_ctx.run
    def check_hello_pytest():
        # check my_var has value from the new context
        assert my_var.get() == (
            expect := hello("Pytest")
        ), f"expected {expect!r}, saw {my_var.get()!r}"

        # MySettings.constrct() shows that
        # the default_factory reflects my_var's new value
        construct = MySettings.construct()
        assert (
            construct.var == my_var.get()
        ), f"expected match, eval'd `{construct.var!r} == {my_var.get()!r}`"

        # environment variable is used to set MySettings().var
        env_var = MySettings.__fields__["var"].field_info.extra["env"]
        environ[env_var.upper()] = hello("Environment")

        # environ has priority over default_factory
        settings = MySettings()
        assert settings.var == (
            expect := hello("Environment")
        ), f"expected {expect!r}, saw {settings.var!r}"

        return True

    # verify decorator called check_hello_pytest()
    # and assigned return value to function's name 
    assert check_hello_pytest == True, "the decorator is not working as intended"

    # if we get this far, the test passes!
    print("Success!")
