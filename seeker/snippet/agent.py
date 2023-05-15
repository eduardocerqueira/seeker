#date: 2023-05-15T16:57:31Z
#url: https://api.github.com/gists/48982daa94445dd136660e57db4ff495
#owner: https://api.github.com/users/Zilize

import os
import openai


class Agent:
    def __init__(self, model, prompt_path, threshold):
        self.model = model
        self.system_prompt = open(prompt_path, 'r').read()
        self.threshold = threshold
        self.messages = list()
        self.counter = [0]
        self.save_file = None

    def create_response(self):
        response_entry = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages
        )
        response = response_entry['choices'][0]['message']['content']
        self.messages.append({"role": "assistant", "content": response})
        self.counter.append(response_entry['usage']['total_tokens'] - sum(self.counter))
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********"_ "**********"e "**********"n "**********"t "**********"r "**********"y "**********"[ "**********"' "**********"u "**********"s "**********"a "**********"g "**********"e "**********"' "**********"] "**********"[ "**********"' "**********"t "**********"o "**********"t "**********"a "**********"l "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"' "**********"] "**********"  "**********"> "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"t "**********"h "**********"r "**********"e "**********"s "**********"h "**********"o "**********"l "**********"d "**********": "**********"
            excess = "**********"
            while excess > 0:
                excess -= self.counter[2]
                del self.counter[2]
                del self.messages[3]
                del self.messages[3]
        return response

    def chat(self, input_message):
        self.messages.append({"role": "user", "content": input_message})
        return self.create_response()

    def reset(self):
        self.save_file = None
        self.messages.clear()
        self.counter.clear()
        self.counter.append(0)

    def launch(self, save=False):
        if save:
            self.save_file = open("record.log", 'w')
            self.save_file.write(f"SYSTEM_PROMPT: {self.system_prompt}\n")
        self.messages.append({"role": "system", "content": self.system_prompt})
        while True:
            input_message = input(">> ")
            if input_message == "q" or input_message == "Q":
                break
            response = self.chat(input_message)
            print(f"$$ {response}")
            if save:
                self.save_file.write(f"USER: {input_message}\n")
                self.save_file.write(f"AGENT: {response}\n")
        if save:
            self.save_file.close()


if __name__ == '__main__':
    openai.api_key = os.getenv("OPENAI_API_KEY")
    agent = Agent(model="gpt-3.5-turbo", prompt_path="agent_prompt.txt", threshold=3500)
    agent.launch(save=True)
