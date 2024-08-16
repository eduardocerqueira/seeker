#date: 2024-08-16T16:56:27Z
#url: https://api.github.com/gists/d5dcb33f1df0b6e79e5733b3198f7bad
#owner: https://api.github.com/users/maciejmajek

    def handle_human_message(self, msg: String): # a callback
        self.get_logger().info("Handling human message")

        # handle human message
        self.history.append(HumanMessage(content=msg.data))
        llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)

        chunks: List[str] = []
        for chunk in llm.stream(self.history):
            if not isinstance(chunk.content, str):
                self.get_logger().error(f"Invalid chunk: {chunk}")
                continue
            if chunk.content:
                chunks.append(chunk.content)
            if chunk.content.endswith((".", "!", "?")) or len(chunks) > 50:
                self.hmi_publisher.publish(String(data=''.join(chunks)))
                chunks = []
        if len(chunks):
            self.hmi_publisher.publish(String(data=''.join(chunks)))
