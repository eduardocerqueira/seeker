#date: 2024-12-30T16:51:29Z
#url: https://api.github.com/gists/3d84b70a9df0e71048b69339d130aa62
#owner: https://api.github.com/users/skrawcz

import pytest

@pytest.fixture(scope="module")
def database_connection():
    """Fixture that creates a DB connection"""
    db_client = SomeDBClient()
    yield db_client
    print("\nStopped client:\n")

def test_my_function(database_connection):
    """pytest will inject the result of the 'database_connection' function 
    into `database_connection` here in this test function"""
    ...

def test_my_other_function(database_connection):
    """pytest will inject the result of the 'database_connection' function 
    into `database_connection` here in this test function"""
    ...