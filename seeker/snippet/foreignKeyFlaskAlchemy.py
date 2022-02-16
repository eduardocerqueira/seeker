#date: 2022-02-16T17:10:16Z
#url: https://api.github.com/gists/f0b539d3dc5890472a8c8d956454002a
#owner: https://api.github.com/users/uhuru-rawlings

from flask import Flask, jsonify, request, make_response
from flask.ext.httpauth import HTTPBasicAuth
from flask_sqlalchemy import SQLAlchemy
from flask import render_template, redirect, url_for
from sqlalchemy import UniqueConstraint, exc

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///memes.db'
app.config['SECRET_KEY'] = 'HALO'

db = SQLAlchemy(app)
auth = HTTPBasicAuth()

STRING_FAIL = 'fail'
STRING_SUCCESS = 'success'
STRING_UPVOTE = 'upvote'
STRING_DOWNVOTE = 'downvote'

class Photo(db.Model):
	__tablename__ = 'photo'
	id = db.Column(db.Integer, primary_key = True, autoincrement = True)
	urlPrimary = db.Column(db.String, nullable = False, unique = True)
	urlSecondary = db.Column(db.String, nullable = True)
	title = db.Column(db.String, nullable = True)
	uploadDate = db.Column(db.String, nullable = False)
	votes = db.relationship('Votes', backref = 'photo', cascade = 'all, delete-orphan', lazy = 'dynamic')

	def __init__ (self,urlPrimary,urlSecondary,title,uploadDate):
		self.urlPrimary = urlPrimary
		self.urlSecondary = urlSecondary
		self.title = title
		self.uploadDate = uploadDate

	def toString(self):
		up = len([v for v in self.votes if v.voteType == STRING_UPVOTE]) # count total upvotes 
		down = len([v for v in self.votes if v.voteType == STRING_DOWNVOTE]) # count total downvotes
		return ({'urlPrimary' : self.urlPrimary, 'urlSecondary': self.urlSecondary,'title':self.title, 'uploadDate':self.uploadDate, 'voteCountUp':up, 'voteCountDown' : down})
    
class Users(db.Model):
	__tablename__ = 'user'
	id = db.Column(db.Integer, primary_key = True, autoincrement = True)
	name = db.Column(db.String, nullable = False)
	email = db.Column(db.String,nullable = False, unique = True)
	registeredDate = db.Column(db.String, nullable = False)
	votes = db.relationship('Votes', backref = 'user', cascade = 'all, delete-orphan', lazy = 'dynamic')

	def __init__(self,name,email,registeredDate):
		self.name = name
		self.email = email
		self.registeredDate = registeredDate

	def toString(self):
		voted = [vote.toString() for vote in self.votes]
		return ({'name':self.name, 'email':self.email, 'registeredDate':self.registeredDate, 'votes' : voted})

class Votes(db.Model):
	__tablename__ = 'vote'
	__table_args__ = (UniqueConstraint('photoID','userID', name = 'unique_photo_user'),)
	id = db.Column(db.Integer, primary_key = True, autoincrement = True)
	photoID = db.Column(db.Integer, db.ForeignKey('photo.id'), nullable = False)
	userID = db.Column(db.Integer, db.ForeignKey('user.id'), nullable = False)
	voteType = db.Column(db.String, nullable = False)
	votedDate = db.Column(db.String, nullable = False)

	def __init__(self,photoID, userID, voteType, votedDate):
		self.photoID = photoID
		self.userID = userID
		self.voteType = voteType
		self.votedDate = votedDate

	def toString(self):
		return ({'photoID':self.photoID, 'userID':self.userID, 'voteType':self.voteType, 'votedDate':self.votedDate})

@app.route('/')
def home():
	return 'The Memions'

@app.route('/photos')
def photoAll():
	photos = Photo.query.all()
	p = [photo.toString() for photo in photos]
	return jsonify(meta = STRING_SUCCESS, photos = p)

@app.route('/photos/new', methods = ['POST'])
def newPhoto():
	jsonData = request.json
	newPhoto = Photo(jsonData['urlPrimary'], jsonData['urlSecondary'], jsonData['title'], jsonData['uploadDate'])
	db.session.add(newPhoto)
	session_commit()
	
@app.route('/photo/<photoid>')
def photoIndividual(photoid):
	photo = Photo.query.filter_by(id = photoid).first()
	if photo is None : 
		return jsonify(meta = STRING_FAIL)
	else:
		return jsonify(meta = STRING_SUCCESS, photo = photo.toString())

@app.route('/users')
def users():
	users = Users.query.all()
	u = [user.toString() for user in users]
	return jsonify(meta = STRING_SUCCESS, users = u)

@app.route('/users/new', methods = ['POST'])
def newUser():
	jsonData = request.json
	newUser = Users(jsonData['name'], jsonData['email'], jsonData['registeredDate'])
	db.session.add(newUser)
	session_commit()

@app.route('/user/<userid>')
def userIndividual(userid):
	user = Users.query.filter_by(id = userid).first()
	if user is None :
		return jsonify(meta = STRING_FAIL)
	else:
		return jsonify(meta = STRING_SUCCESS, user = user.toString())

@app.route('/votes')
def votes():
	votes = Votes.query.all()
	v = [vote.toString() for vote in votes]
	return jsonify(meta = STRING_SUCCESS, votes = v)

@app.route('/votes/new', methods = ['POST'])
def newVotes():
	jsonData = request.json
	newVote = Votes(jsonData['photoID'], jsonData['userID'], jsonData['voteType'], jsonData['votedDate'])
	db.session.add(newUser)
	session_commit()
	
def session_commit():
	try :
		db.session.commit()
		return jsonify(meta = STRING_SUCCESS)
	except exc.IntegrityError:
		print ("IntegrityError while adding new user")
		db.session.rollback()
		return jsonify(meta = STRING_FAIL)

if __name__ == '__main__' :
	db.create_all()
	"""
	rish = Users(name = 'rish', email ='rish@gmail.com', registeredDate = '123456789')
	meme = Photo( urlPrimary = 'urlPrimary', urlSecondary = 'urlSecondary', title = 'memeTitle', uploadDate ='1234567898')
	voteup = Votes(1,1, voteType = STRING_UPVOTE, votedDate = '98765432')
	db.session.add(rish)
	db.session.add(meme)
	db.session.add(voteup)
	try:
		db.session.commit()
	except exc.IntegrityError: 
		print 'IntegrityError while committing brah'
		db.session.rollback()

	print Photo.query.get(1).votes.count()
	"""
	app.run(debug=True)