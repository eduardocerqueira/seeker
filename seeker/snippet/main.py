#date: 2022-08-25T16:47:01Z
#url: https://api.github.com/gists/dc754147d29efffb583795b8d3da21e6
#owner: https://api.github.com/users/Utkarshsingh001

## MORE CODE ABOVE

@app.route('/register', methods=["GET", "POST"])
def register():
    form = RegisterForm()
    if form.validate_on_submit():

        hash_and_salted_password = "**********"
            form.password.data,
            method='pbkdf2:sha256',
            salt_length=8
        )
        new_user = User(
            email=form.email.data,
            name=form.name.data,
            password= "**********"
        )
        db.session.add(new_user)
        db.session.commit()
        
        #This line will authenticate the user with Flask-Login
        login_user(new_user)
        return redirect(url_for("get_all_posts"))

    return render_template("register.html", form=form)
  
  ## MORE CODE BELOW  
  ## MORE CODE BELOW