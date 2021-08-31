#date: 2021-08-31T02:50:28Z
#url: https://api.github.com/gists/9146014546155de74198c876b2e8ac7b
#owner: https://api.github.com/users/Codethier

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        id: int = payload.get("id")
        if id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.get(User, id)
    if user is None:
        raise credentials_exception
    return user


async def check_if_admin(user: schemas.User = Depends(get_current_user)):
    if user.role == "admin":
        return user
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is not admin",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def authenticate_user(db: Session, username: str, password: str):
    try:
        user: User = db.execute(select(User).filter(User.username == username).filter(User.password == password)).scalar_one()
        data_dict = {"id": user.user_id, "username": user.username, "role": user.role}
        return create_access_token(data_dict)
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.post('/token')
async def token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    token = authenticate_user(db, form_data.username, form_data.password)
    return {"access_token": token, "token_type": "bearer"}
    
    
and an admin route 

@app.post("/admin/clothes/clothe_type/create", dependencies=[Depends(check_if_admin)])
def create_clothe_type(obj: schemas.TypeBase, db: Session = Depends(get_db)):
    clothe_type = type_dict_maker(clothe_type=obj.clothe_type)
    commit_and_save(db, clothe_type)
    pk = clothe_type.type_id
    return pk