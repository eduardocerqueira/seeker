#date: 2025-01-09T16:56:46Z
#url: https://api.github.com/gists/4939a84f9c476333d517de466e1b8ee0
#owner: https://api.github.com/users/ishii-kanade

import minescript
import sys
from openai import OpenAI
from datetime import datetime
import os

def fetch_chatgpt_response(api_key, px, py, pz, user_content):
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "あなたは優秀なマインクラフターです。"},
                {"role": "system", "content": f"あなたのプレイヤーは、x座標:{int(px)}、y座標:{int(py)}、z座標:{int(pz)} の場所にいます。"},
                {"role": "system", "content": "回答はPython言語で、配置ブロックの(x座標(整数),y座標(整数),z座標(整数),ブロックID(文字列))のタプルの集合を返すchatgpt_work()関数のみを定義してください。"},
                {"role": "system", "content": "回答はできるかぎりfor文やwhile文などを使って効率化してください。"},
                {"role": "system", "content": "回答文にはコメント文を含めないでください。"},
                {"role": "system", "content": "ブロックIDは質問に対して最適なマインクラフト上のブロックID(oak_logなど)を返してください。"},
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        minescript.echo(f"Error in OpenAI API: {e}")
        sys.exit(1)

def export_code_to_file(code, directory="XXXXXXX"):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"generated_script_{timestamp}.py"
        file_path = os.path.join(directory, file_name)
        code = code.replace("```python\n", "").replace("```", "")
        with open(file_path, "w") as f:
            f.write(code)
        minescript.echo(f"Code exported to {file_path}")
        return file_path
    except Exception as e:
        minescript.echo(f"Error exporting code: {e}")
        sys.exit(1)

def execute_code_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            exec(f.read(), globals())
    except Exception as e:
        minescript.echo(f"Error executing code from file: {e}")
        sys.exit(1)

def place_blocks(block_data):
    blockpacker = minescript.BlockPacker()
    for d in block_data:
        block_id = d[3]
        blockpacker.setblock((int(d[0]), int(d[1]), int(d[2])), block_id)
    blockpacker.pack().write_world()

if __name__ == "__main__":
    (px, py, pz) = minescript.player_position()
    user_content = sys.argv[1]
    api_key = "XXXXXXXXXX"
    response = fetch_chatgpt_response(api_key, px, py, pz, user_content)
    exported_file_path = export_code_to_file(response)
    execute_code_from_file(exported_file_path)
    try:
        if "chatgpt_work" not in globals():
            raise NameError("chatgpt_work is not defined in the generated code.")
        block_data = chatgpt_work()
        place_blocks(block_data)
    except Exception as e:
        minescript.echo(f"Error in block placement: {e}")
        sys.exit(1)
