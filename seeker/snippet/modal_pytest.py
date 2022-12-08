#date: 2022-12-08T17:05:50Z
#url: https://api.github.com/gists/b41501b4a26229024e2842d1c1729bf2
#owner: https://api.github.com/users/helloworld

import modal

pytest_image = modal.Image.debian_slim().pip_install(["pytest"])
stub = modal.Stub()

code = """
def hello_world(name=None, age=None, city=None):
    return 'Hello World!'


def test_hello_world():
    assert hello_world() == 'Hello World!'


def test_hello_world_with_name():
    assert hello_world('John') == 'Hello John!'
    assert hello_world(name='John') == 'Hello John!'


def test_hello_world_with_name_and_age():
    assert hello_world('John', 30) == 'Hello John! You are 30 years old.'
    assert hello_world(
        name='John', age=30) == 'Hello John! You are 30 years old.'


def test_hello_world_with_name_and_age_and_city():
    assert hello_world(
        'John', 30, 'New York') == 'Hello John! You are 30 years old. You are from New York.'
    assert hello_world(
        name='John', age=30, city='New York') == 'Hello John! You are 30 years old. You are from New York.'
    """


@stub.function(image=pytest_image, shared_volumes={"/root/pytest": modal.SharedVolume()})
def pytest_runner():
    import tempfile
    import subprocess

    with tempfile.TemporaryDirectory() as dirname:
        with open(f"{dirname}/test.py", "w") as f:
            f.write(code)

        try:
            output = subprocess.check_output(
                ["pytest", "test.py"], cwd=dirname, stderr=subprocess.STDOUT
            ).decode("utf-8")
        except subprocess.CalledProcessError as e:
            output = e.output.decode("utf-8")

        print(output)


if __name__ == "__main__":
    with stub.run():
        pytest_runner()
