#date: 2023-07-28T16:51:05Z
#url: https://api.github.com/gists/8e79a06f15d93bd9915f1309dddc7913
#owner: https://api.github.com/users/JoelBender

import sys
import os
import os.path

import streamlit as st
from streamlit.logger import get_logger

# humph
# utils_path = os.path.realpath(os.path.pardir)
# if utils_path not in sys.path:
#     sys.path.insert(1, os.path.realpath(os.path.pardir))

from ..utils import load_graph

LOGGER = get_logger(__name__)


def run():
    LOGGER.debug("run")

    st.set_page_config(
        page_title="Info",
    )

    st.write("# Graph Information")

    graph = load_graph()
    st.write(graph)

if __name__ == "__main__":
    run()
