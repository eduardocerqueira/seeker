//date: 2021-12-10T17:09:02Z
//url: https://api.github.com/gists/da7a6788d5602750bdf4b1c5a895fb25
//owner: https://api.github.com/users/magurofly


/*
LL 法は左再帰が苦手なので、構文の形を書き直す必要がある

これを
<expression> ::= <term> | <expression> "+" <term> | <expression> "-" <term>
<term> ::= <factor> | <term> "*" <factor>
<factor> ::= "R" | "S" | "P" | "?" | "(" <expression> ")"

こうする
<expression> ::= <term> ("+" <term> | "-" term)*
<term> ::= <factor> ("*" <factor>)*
<factor> ::= "R" | "S" | "P" | "?" | "(" <expression> ")"

使い方:
var parser = new Parser(入力文字列);
var result = parser.expression(0).value;
 */

class Parser {
    String input;
  
    // 結果は読み込んだ文字数と値を持つ
    class Result<T> {
        public final int consumed;
        public final T result;
        public Result(int consumed, T result) {
            this.consumed = consumed;
            this.result = result;
        }
    }
    
    public Parser(String input) {
        this.input = input;
    }
  
    // 読み込みを開始したいインデックスを渡す
    // 失敗したら null を返す
    public Result<int[]> expression(int index) {
        int consumed = 0;
        
        var x = term(index + consumed); // term を読む
        if (x == null) return null; // term が失敗したら expression も失敗
        // この時点で expression は成功なので、 index を進める
        consumed += x.consumed;
      
        var value = x.value;
        
        for (;;) {
            if (input.charAt(index) == '+') { // "+" <term>
                // ここで "+" を読み込んだが、まだ consumed をすすめてはいけない　なぜなら、次の <term> が失敗したらそれも失敗するから
                var y = term(index + consumed + 1);
                if (y == null) break;
                consumed += 1 + y.consumed;
                value = add(value, y.value);
            } else if (input.charAt(index) == '-') { // "-" <term>
                var y = term(index + consumed);
                if (y == null) break;
                consumed += 1 + y.consumed;
                value = subtract(value, y.index);
            } else {
                break;
            }
        }
        
        return new Result(consumed, value);
    }
}