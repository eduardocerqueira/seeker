#date: 2022-02-09T16:58:27Z
#url: https://api.github.com/gists/e0784646c0e489a39ad14370ca48f8f0
#owner: https://api.github.com/users/arnotae

from sqlalchemy import event
from sqlalchemy.orm import sessionmaker
from app.db.session import engine
import pytest
import app.tests.config


@pytest.fixture(
    scope='function',
    autouse=True  # New test DB session for each test todo we need it only for tests with Client fixture
)
def db():
    """
    SQLAlchemy session started with SAVEPOINT
    After test rollback to this SAVEPOINT
    """
    connection = engine(
        app.tests.config.get_test_config()
    ).connect()

    # begin a non-ORM transaction
    trans = connection.begin()
    session = sessionmaker()(bind=connection)

    session.begin_nested()  # SAVEPOINT

    app.tests.config.session = session  # Inject session to the server code under test

    @event.listens_for(app.tests.config.session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        """
        Each time that SAVEPOINT ends, reopen it
        """
        if transaction.nested and not transaction._parent.nested:
            session.begin_nested()

    yield session

    session.close()
    trans.rollback()  # roll back to the SAVEPOINT
    connection.close()