#date: 2023-07-12T17:08:32Z
#url: https://api.github.com/gists/991cf6ec392a5f4b9d61c6e7275bfe63
#owner: https://api.github.com/users/vizovitin

#!/usr/bin/env -S streamlit run
# Requirements: Python>=3.10 streamlit==1.24.0 slack-sdk==3.21.3
import logging
import os
import urllib.parse
import datetime

import slack_sdk
import streamlit as st

logging.basicConfig(level=logging.DEBUG)


@st.cache_resource
 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"l "**********"a "**********"c "**********"k "**********"_ "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"( "**********"b "**********"o "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    return slack_sdk.WebClient(bot_token)


@st.cache_data
def url_to_query_dict(url):
    parsed = urllib.parse.urlparse(url)
    assert parsed.netloc.endswith(".slack.com")
    query_parts = parsed.path.lstrip('/').split('/')
    query_params = urllib.parse.parse_qs(parsed.query)
    assert query_parts[0] == "archives"

    channel_id = query_parts[1]
    if query_params.get('cid'):
        channel_id = query_params['cid'][-1]

    assert query_parts[2].startswith("p")
    message_id = f"{query_parts[2][1:-6]}.{query_parts[2][-6:]}"

    thread_id = None
    if query_params.get('thread_ts'):
        thread_id = query_params['thread_ts'][-1]

    return {
        'cid': channel_id,
        'ts': message_id,
        'thread_ts': thread_id,
    }


def date_to_slack_ts(date):
    return str(datetime.datetime.combine(date, datetime.time()).timestamp())


st.title("Slack Web API demo")

setup_tab, message_tab, history_tab, conversation_tab = st.tabs(["Setup", "Message", "History", "Conversation"])

with setup_tab:
    st.subheader("API tokens")
    bot_token = "**********"="password", value=os.environ.get("SLACK_BOT_TOKEN", ""))
    client = "**********"

with message_tab:
    st.subheader("Message")
    url = st.text_input("Slack message URL")
    if url:
        with st.expander("message parameters"):
            params = url_to_query_dict(url)
            st.json(params)

        with st.expander("conversations_replies()"):
            messages = client.conversations_replies(
                channel=params['cid'],
                ts=params['ts'],
                oldest=params['ts'],
                latest=params['ts'],
                inclusive=True,
                limit=1,
                include_all_metadata=True,
            )
            st.json(messages.data)

        with st.expander("reactions_get()"):
            reactions = client.reactions_get(
                channel=params['cid'],
                timestamp=params['ts'],
            )
            st.json(reactions.data)

        with st.expander("users_info()"):
            users = client.users_info(user=messages['messages'][0].get('user'))
            st.json(users.data)

with history_tab:
    st.subheader("History")
    columns = st.columns([3, 1, 1])
    cid = columns[0].text_input("Channel ID or URL", key="history-channel").split('/')[-1]
    ts_from = date_to_slack_ts(columns[1].date_input("From date"))
    ts_to = date_to_slack_ts(columns[2].date_input("To date"))
    if cid:
        for message in client.conversations_history(channel=cid, oldest=ts_from, latest=ts_to):
            st.json(message.data)

with conversation_tab:
    st.subheader("Conversation")
    columns = st.columns([4, 1])
    cid = columns[0].text_input("Channel ID or URL", key="conversation-channel").split('/')[-1]

    if cid and columns[1].button("Join", use_container_width=True):
        reply = client.conversations_join(channel=cid)
        st.json(reply.data)

    if cid and columns[1].button("Leave", use_container_width=True):
        # This is typically not available for bots
        reply = client.conversations_leave(channel=cid)
        st.json(reply.data)

    if cid:
        with st.expander("conversations_info()"):
            info = client.conversations_info(
                channel=cid,
                include_locale=True,
                include_num_members=True,
            )
            st.json(info.data)

        with st.expander("conversations_members()"):
            members = client.conversations_members(channel=cid)
            st.json(members.data)
