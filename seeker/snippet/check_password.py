#date: 2025-02-05T17:06:57Z
#url: https://api.github.com/gists/8a557fa8152958c27485accdf409a23e
#owner: https://api.github.com/users/djun

 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"h "**********"e "**********"c "**********"k "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"( "**********") "**********": "**********"
    """Returns `True` if the user had a correct password."""

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"e "**********"n "**********"t "**********"e "**********"r "**********"e "**********"d "**********"( "**********") "**********": "**********"
        """Checks whether a password entered by the user is correct."""
        user = authenticate(
            username=st.session_state['username'], 
            password= "**********"
            )
        
        if (user is not None):
            st.session_state["password_correct"] = "**********"
            del st.session_state["password"]  # don't store username + password
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"c "**********"o "**********"r "**********"r "**********"e "**********"c "**********"t "**********"" "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"n "**********"  "**********"s "**********"t "**********". "**********"s "**********"e "**********"s "**********"s "**********"i "**********"o "**********"n "**********"_ "**********"s "**********"t "**********"a "**********"t "**********"e "**********": "**********"
        # First run, show inputs for username + password.
        st.text_input("Username", on_change= "**********"="username")
        st.text_input(
            "Password", type= "**********"=password_entered, key="password"
        )
        return False
 "**********"  "**********"  "**********"  "**********"  "**********"e "**********"l "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"t "**********". "**********"s "**********"e "**********"s "**********"s "**********"i "**********"o "**********"n "**********"_ "**********"s "**********"t "**********"a "**********"t "**********"e "**********"[ "**********"" "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"c "**********"o "**********"r "**********"r "**********"e "**********"c "**********"t "**********"" "**********"] "**********": "**********"
        # Password not correct, show input + error.
        st.text_input("Username", on_change= "**********"="username")
        st.text_input(
            "Password", type= "**********"=password_entered, key="password"
        )
        st.error("😕 User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True