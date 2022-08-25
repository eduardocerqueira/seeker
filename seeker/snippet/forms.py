#date: 2022-08-25T17:06:54Z
#url: https://api.github.com/gists/b2cc682c8870b260c2cc12503e713e97
#owner: https://api.github.com/users/Utkarshsingh001

## MORE CODE ABOVE

class CommentForm(FlaskForm):
    comment_text = CKEditorField("Comment", validators=[DataRequired()])
    submit = SubmitField("Submit Comment")
    
## MORE CODE BELOW