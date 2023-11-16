#date: 2023-11-16T16:40:21Z
#url: https://api.github.com/gists/1bb2efd416c858bee3722e56c2568b4b
#owner: https://api.github.com/users/FernandoCelmer

import pytest

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
from sqlalchemy.exc import IntegrityError

Base = declarative_base()


class Author(Base):
    __tablename__ = 'authors'

    id = Column(Integer(), primary_key=True)
    firstname = Column(String(100))
    lastname = Column(String(100))
    email = Column(String(255), nullable=False)
    joined = Column(DateTime(), default=datetime.now)
    articles = relationship('Article', backref='author')


class Article(Base):
    __tablename__ = 'articles'

    id = Column(Integer(), primary_key=True)
    slug = Column(String(100), nullable=False)
    title = Column(String(100), nullable=False)
    created_on = Column(DateTime(), default=datetime.now)
    updated_on = Column(DateTime(), default=datetime.now, onupdate=datetime.now)
    content = Column(Text)
    author_id = Column(Integer(), ForeignKey('authors.id'))

engine = create_engine("sqlite:///./sql_app.db")
Session = sessionmaker(bind=engine)


# ========= Tests ==========

class TestBlog:

    def setup_class(self):
        Base.metadata.create_all(engine)
        self.session = Session()
        self.valid_author = Author(
            firstname="Ezzeddin",
            lastname="Aybak",
            email="aybak_email@gmail.com"
        )

    def teardown_class(self):
        self.session.rollback()
        self.session.close()

    def test_author_valid(self):   
        self.session.add(self.valid_author)
        self.session.commit()
        aybak = self.session.query(Author).filter_by(lastname="Aybak").first()
        assert aybak.firstname == "Ezzeddin"
        assert aybak.lastname != "Abdullah"
        assert aybak.email == "aybak_email@gmail.com"

    @pytest.mark.xfail(raises=IntegrityError)
    def test_author_no_email(self):
        author = Author(
            firstname="James",
            lastname="Clear"
        )
        self.session.add(author)
        try:
            self.session.commit()
        except IntegrityError:
            self.session.rollback()

    def test_article_valid(self):
        valid_article = Article(
            slug="sample-slug",
            title="Title of the Valid Article",
            content="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
            author=self.valid_author
            )
        self.session.add(valid_article)
        self.session.commit()
        sample_article = self.session.query(Article).filter_by(slug="sample-slug").first()
        assert sample_article.title == "Title of the Valid Article"
        assert len(sample_article.content.split(" ")) > 50
