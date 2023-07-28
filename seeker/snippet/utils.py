#date: 2023-07-28T16:51:05Z
#url: https://api.github.com/gists/8e79a06f15d93bd9915f1309dddc7913
#owner: https://api.github.com/users/JoelBender

import sys

import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)


@st.cache_resource
def load_graph():
    LOGGER.debug("load_graph")

    return "just a thing"

