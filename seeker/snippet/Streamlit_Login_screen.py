#date: 2021-12-20T17:13:00Z
#url: https://api.github.com/gists/d3f81ef314094c52509935139b8c6cba
#owner: https://api.github.com/users/dormeir999

# Helper fucntions

def is_authenticated(user, password):
    if (user == 'Admin' and password == "12345678"):
        return True
    else:
        return False
def linespace_generator(n_spaces=1):
    for i in range(n_spaces):
        st.write("")
        
# Implementation, should be used in the end of if __name__ == "__main__":

if __name__ == "__main__":
 ...
 if not state.authenticated:
   col3, col4, col5, col6 = st.columns(4)
   col4.title("Awesome App")
   col4.write('Brought to you by Great Company')
   state.user = col5.text_input('User', key='user', value="")
   state.password = col5.text_input('Password', type="password", value="")
   if state.user and state.password:
      login_button = col5.button("Login")
   state.authenticated = is_authenticated(state.user, state.password) and login_button
   state.sync()
   if 'login_button' not in locals():
      login_button = False
   if login_button and state.user and state.password and not state.authenticated:
      col5.info("Please check your credentials")
 else:
  main(state)