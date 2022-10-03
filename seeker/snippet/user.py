#date: 2022-10-03T17:08:49Z
#url: https://api.github.com/gists/0a122619ca442a16608a0449ba769305
#owner: https://api.github.com/users/aideedprogrammer

@bp_user.route('/list', methods=['GET'])
def list():
    response = dict()
    response["status"] = "OK"
    response["data"] = []
    response["error"] = "-"

    try:
        user = User.query.all()
        if user:
            for i in user:
                item = dict()
                item['name'] = i.name
                item['age'] = i.age
                item['birth'] = i.date_birth
                item['gender'] = i.gender

            response["data"].append(item)

    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        msg = exc_obj, fname, "Line number : ", exc_tb.tb_lineno
        print(msg)

        db.session.rollback()
        
        response["status"] = "Fail"
        response["error"] = str(msg)
    
    finally:
        return jsonify(response)


@bp_user.route('/add', methods=['POST'])
def add():
    response = dict()
    response["status"] = "OK"
    response["data"] = []
    response["error"] = "-"

    try:
        data = request.form['data']
        items = json.loads(data)

        user = User(name=items['name'],age=items['age'],date_birth=items['date_birth'],gender=items['gender'])
        db.session.add(user)

    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        msg = exc_obj, fname, "Line number : ", exc_tb.tb_lineno
        print(msg)

        db.session.rollback()

        response["status"] = "Fail"
        response["error"] = str(msg)

    finally:
        db.session.commit()

        return jsonify(response)


@bp_user.route('/update', methods=['POST'])
def update():
    response = dict()
    response["status"] = "OK"
    response["data"] = []
    response["error"] = "-"

    try:
        data = request.form['data']
        items = json.loads(data)

        user = User.query.filter_by(name=items['previous_name']).first()
        if user:
            user.name = items['name']
            user.age = items['age']
            user.gender = items['gender']

            db.session.add(user)

    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        msg = exc_obj, fname, "Line number : ", exc_tb.tb_lineno
        print(msg)

        db.session.rollback()

        response["status"] = "Fail"
        response["error"] = str(msg)

    finally:
        db.session.commit()

        return jsonify(response)


@bp_user.route('/delete', methods=['POST'])
def delete():
    response = dict()
    response["status"] = "OK"
    response["data"] = []
    response["error"] = "-"

    try:
        data = request.form['data']
        items = json.loads(data)

        user = User.query.filter_by(name=items['name']).first()
        if user:
            db.session.delete(user)

    except:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        msg = exc_obj, fname, "Line number : ", exc_tb.tb_lineno
        print(msg)

        db.session.rollback()

        response["status"] = "Fail"
        response["error"] = str(msg)

    finally:
        db.session.commit()

        return jsonify(response)


def calculatePeople():
    total = 5 + 2
    return total