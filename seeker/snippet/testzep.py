#date: 2025-02-10T17:01:47Z
#url: https://api.github.com/gists/32c4936c56e46dd5ccccf9aa99e248e5
#owner: https://api.github.com/users/boat-builder


user = "testuser1"
agent = "berlin-ecom1"
user_session_id = "60d86e48-b48f-4b9c-909f-5c248bd7ea73"
agent_session_id = "207a4a3a-a5be-4dc1-9b42-2c5ddfa6d5c6"


async def main():
    resp = await zep.user.add(user_id=user)
    resp = await zep.user.add(user_id=agent)
    resp = await zep.memory.add_session(user_id=user, session_id=user_session_id)
    resp = await zep.memory.add_session(user_id=agent, session_id=agent_session_id)

    resp = await zep.memory.add(
        session_id=user_session_id,
        messages=[
            Message(
                role_type="user",
                content="Your suggestion contains schlage product. But we don't sell anything other than sargent products.",
            ),
        ],
    )
    print("user message added", resp)
    resp = await zep.memory.add(
        session_id=agent_session_id,
        messages=[
            Message(
                role_type="assistant",
                content="latest SEO knowledge in ecommerce is to create product pages over category and blog/article pages. Product pages ranks better",
            )
        ],
    )
    print("agent message added", resp)
    fact_response = await zep.user.get_facts(user_id=user)
    print("user facts", fact_response)
    fact_response = await zep.user.get_facts(user_id=agent)
    print("agent facts", fact_response)
    r = await zep.graph.search(
        user_id=user,
        query="I just found three toics to create a page for SEO. Which one should I pick and what page should I make? \n1. Schlage hardware and doors\n2. Sargent door locks\n3. Van Du Prin door locks",
        limit=4,
        scope="edges",
    )
    print("user search", r)
    r = await zep.graph.search(
        user_id=agent,
        query="I just found three toics to create a page for SEO. Which one should I pick and what page should I make? \n1. Schlage hardware and doors\n2. Sargent door locks\n3. Van Du Prin door locks",
        limit=4,
        scope="edges",
    )
    print("agent search", r)


if __name__ == "__main__":
    asyncio.run(main())
