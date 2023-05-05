#date: 2023-05-05T16:44:31Z
#url: https://api.github.com/gists/115bd219324caf258e60cc90ba0384a2
#owner: https://api.github.com/users/ayratKEK777

import logging
from flask import Flask, render_template, url_for, redirect
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SubmitField, SelectField, FileField
from wtforms.validators import DataRequired, Length
from flask_wtf.file import FileRequired, FileAllowed
from werkzeug.utils import secure_filename
from pathlib import Path


BASEDIR = Path(__file__).parent
UPLOAD_FOLDER = BASEDIR / 'static' / 'images'

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config["SECRET_KEY"] = "**********"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite3"

db = SQLAlchemy(app)


class Movie(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), unique=True, nullable=False)
    text = db.Column(db.Text, nullable=False)
    image = db.Column(db.String, nullable=True)
    reviews = db.relationship("Reviews", back_populates="movie")


class Reviews(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    text = db.Column(db.Text, nullable=False)
    created_date = db.Column(db.DateTime, default=datetime.now)
    score = db.Column(db.Integer, nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey("movie.id"))
    movie = db.relationship("Movie", back_populates="reviews")


db.create_all()


class MovieForm(FlaskForm):
    title = StringField(
        'Название',
        validators=[DataRequired(message="Поле не должно быть пустым"),
                    Length(max=255, message='Введите заголовок длиной до 255 символов')]
    )
    text = TextAreaField(
        'Описание',
        validators=[DataRequired(message="Поле не должно быть пустым")])

    image = FileField("постер", validators=[FileRequired(message="Добавьте изображение"),
                                                 FileAllowed(["jpg", "img", "png", "jpeg"], message="Неверный формат")])

    submit = SubmitField('Добавить кинцо ')




@app.route("/")
def index():
    movies = Movie.query.all()
    return render_template('index.html', movies=movies)


@app.route("/movie/<int:id>")
def movie(id):
    return render_template('movie.html')


@app.route("/add_movie", methods=["POST", "GET"])
def add_movie():
    form = MovieForm()
    if form.validate_on_submit():
        mv = Movie()
        mv.title = form.title.data
        mv.text = form.text.data
        image = form.image.data
        image_name = secure_filename(image.filename)
        UPLOAD_FOLDER.mkdir(exist_ok=True)
        image.save(UPLOAD_FOLDER/image_name)
        mv.image = image_name
        db.session.add(mv)
        db.session.commit()
        return redirect(url_for("index"))

    return render_template('add_movie.html', form=form)


@app.route("/reviews")
def reviews():
    return render_template('reviews.html')


@app.route("/delete_review/<int:id>")
def delete_review(id):
    return render_template('delete_review.html')


if __name__ == '__main__':
    app.run(debug=True)

ue)

