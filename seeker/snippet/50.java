//date: 2022-03-09T17:10:41Z
//url: https://api.github.com/gists/98de38e3fbdc648d12e55a7f79b4efae
//owner: https://api.github.com/users/TehleelMir

class Solution {
    public String removeKdigits(String num, int k) {
        Stack<Character> stack = new Stack<Character>();
        
        for(int i = 0; i < num.length(); i++) {
            char c = num.charAt(i);
            
            while(!stack.empty() && k > 0 && stack.peek() > c) {
                stack.pop();
                k--;
            }
            
            if(!stack.empty() || c != '0') 
                stack.push(c);
        }
        
        while(!stack.empty() && k-- > 0)
            stack.pop();
        
        if(stack.empty()) 
            return "0";
        
        StringBuilder str = new StringBuilder("");
        for(char temp : stack) 
            str.append(temp);
        
        return str.toString();
    }
}