#date: 2022-01-26T17:03:33Z
#url: https://api.github.com/gists/e0ab366fc0dbc092c1a5946b30caa9a0
#owner: https://api.github.com/users/brunomsantiago

import streamlit as st
import streamlit.components.v1 as components


def left_callback():
    st.write('Left button was clicked')


def right_callback():
    st.write('Right button was clicked')


left_col, right_col, _ = st.columns([1, 1, 3])

with left_col:
    st.button('LEFT', on_click=left_callback)

with right_col:
    st.button('RIGHT', on_click=right_callback)

components.html(
    """
<script>
const doc = window.parent.document;
buttons = Array.from(doc.querySelectorAll('button[kind=primary]'));
const left_button = buttons.find(el => el.innerText === 'LEFT');
const right_button = buttons.find(el => el.innerText === 'RIGHT');
doc.addEventListener('keydown', function(e) {
    switch (e.keyCode) {
        case 37: // (37 = left arrow)
            left_button.click();
            break;
        case 39: // (39 = right arrow)
            right_button.click();
            break;
    }
});
</script>
""",
    height=0,
    width=0,
)
