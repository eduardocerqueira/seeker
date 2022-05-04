#date: 2022-05-04T17:15:35Z
#url: https://api.github.com/gists/41fb117dc1a38098a6392bb577f1dc43
#owner: https://api.github.com/users/diegounzueta

class app:
   def __init__(self):
      ...
      self.pipeline()

   def pipeline(self):
         self.initialize_tool()
         self.buttons()
         self.load_past()
         asyncio.run(self.send_receive())

   def initialize_tool(self):
      st.set_page_config(page_title="Interactive AI", page_icon="🤖")
      st.markdown('<h1 style="color: white">SMART ASSISTANT TOOL</h1>', unsafe_allow_html=True)
      ...
            
   def clear_chat(self):
      ...

   def buttons(self):
      ...

   async def send_receive(self):
         ...
         async def send():
               ...
         async def receive():
               ...

if __name__ == '__main__':
    app()