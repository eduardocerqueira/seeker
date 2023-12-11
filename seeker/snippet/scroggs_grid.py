#date: 2023-12-11T16:54:06Z
#url: https://api.github.com/gists/ded16b1d5d76317f15d53de0148878d6
#owner: https://api.github.com/users/dmadisetti

from IPython.display import HTML, display
import numpy as np
import pandas as pd
from IPython.display import HTML
import pandas as pd
from bs4 import BeautifulSoup
import math

def math_grid(grid, args, ans=None, callback=None):
  style="""<style>
    .grid {
      font-family: sans-serif;
      border-collapse: collapse;
      text-align: center;
      margin-left: auto;
      margin-right: auto;
      border: 0px;
  }
    .bsq{
      text-align: center;
      font-family: sans-serif;
      line-height: 1.4em;
      border-collapse: collapse;
      width: 20px;
      height: 20px;
      border: 1px solid black;
      font-size: 14px;
    }
    .red{
      background-color: #FFBBBB;
      color: #333;
    }
    table.grid td {
        font-size: 16px;
        width: 20px;
        height: 20px;
    }
    input[type='number'] {
      width: 30px;
      height: 30px;
      border: none;
      text-align: center;
    }
    #copyButton {
      margin-top: 10px;
    }
    </style>"""
  def test(args, grid, callback):
    parsed_html = BeautifulSoup(grid, features="lxml")
    tds = parsed_html.body.find_all('td')
    ans = list(map(lambda i:int(tds[i].text.split()[-1].split("=")[-1]), [5,17,29,30,32,34]))
    sq = parsed_html.body.find_all('td', {'class':'bsq'})

    table = [[args[0], tds[1].text, args[1], tds[3].text, args[2]],
            [tds[6].text, "", tds[8].text, "", tds[10].text],
            [args[3], tds[13].text, args[4], tds[15].text, args[5]],
            [tds[18].text, "", tds[20].text, "", tds[22].text],
            [args[6], tds[25].text, args[7], tds[27].text, args[8]]]

    ops = {"+":(lambda a,b:a+b), "×":(lambda a,b:a*b), "–":(lambda a,b:a-b), "÷":(lambda a,b:a/b)}
    order = list(filter(lambda x:x and isinstance(x,str), table[0]+table[2]+table[4]+table[1]+table[3]))
    def op(idx):
      return ops[order[idx]]
    # Rows
    for i in range(3):
      try:
        assert op(i*2+1)(op(i*2)(args[i*3], args[i*3+1]), args[i*3+2]) == ans[i]
      except AssertionError:
        raise AssertionError(f"({args[i*3]} {order[i*2]} {args[i*3+1]}) {order[i*2+1]} {args[i*3+2]} != {ans[i]}")

    # Column
    for i in range(3):
      try:
        assert op(9 + i)(op(6 + i)(args[i], args[3 + i]), args[6 + i]) == ans[3 + i]
      except AssertionError:
        raise AssertionError(f"({args[i]} {order[6 + i]} {args[3 + i]}) {order[9+i]} {args[6 + i]} != {ans[3 + i]}")


    for i, s in zip(args, sq):
      s.string = str(i)

    return HTML(f"answer: {callback([args[i] for s, i in zip(sq, range(9)) if 'red' in s['class']])}"
  +style+repr(parsed_html))
  # HTML(style + grid)

  if len(args):
    if callback == None:
      callback = np.product
    display(test(sum(args,[]), grid, callback))
  else:
    # JavaScript to make the grid editable and add copy functionality
    script = """
    <script>
      (function() {
        // Make bsq cells editable
        var bsqSquares = document.querySelectorAll('.bsq');
        bsqSquares.forEach(function(sq, index) {
          var input = document.createElement('input');
          input.type = 'number';
          input.min = '0';
          input.max = '9';
          sq.innerHTML = '';
          input.className = 'input' + Math.floor(index / 3); // Class name based on row
          sq.appendChild(input);
        });

        function generateArgs() {
          var args = [];
          for (var i = 0; i < 3; i++) {
            var row = [];
            var inputs = document.querySelectorAll('.input' + i);
            inputs.forEach(function(input) {
              row.push(input.value || '0');
            });
            args.push('[' + row.join(', ') + ']');
          }
          return '[' + args.join(', ') + ']';
        }

        // Copy to clipboard
        document.getElementById('copyButton').addEventListener('click', function() {
          var argsText = generateArgs();
          document.getElementById('args').innerHTML=argsText;
          navigator.clipboard.writeText(argsText).then(function() {
            // alert('Copied to clipboard: ' + argsText);
          }, function(err) {
            alert('Error in copying text: ' + err);
          });
        });
      })()
    </script>
    """

    # Button to copy the code
    copy_button = "<button id='copyButton'>Copy Code to Clipboard</button><pre id=args></pre>"

    # Combine everything
    display(HTML(style + grid + script + copy_button))