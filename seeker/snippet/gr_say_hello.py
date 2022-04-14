#date: 2022-04-14T16:59:46Z
#url: https://api.github.com/gists/11bab653c746f7bdaf83980fd2df27e1
#owner: https://api.github.com/users/keitazoumana

# The Hello Function
def say_hello(user):

  return "Hello {} & Welcome to Gradio!".format(user)

# The UI implementation
iface = gr.Interface(fn=say_hello, inputs="text", outputs="text")
iface.launch()