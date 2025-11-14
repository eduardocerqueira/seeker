//date: 2025-11-14T17:09:25Z
//url: https://api.github.com/gists/35131abc877da523fc248c156d67a01d
//owner: https://api.github.com/users/aCoderOfTheSevenKingdoms

class TrieNode {
   
   private TrieNode[] links;
   private boolean flag;

   public TrieNode(){
    this.links = new TrieNode[26];
    this.flag = false;
   }

   // Checks if the current node contains an alphabet
   boolean containsKey(char ch){
    return (this.links[ch - 'a'] != null);
   }
   
   // Add new reference of a new alphabet in the current trie
   void put(char ch, TrieNode node){
    links[ch - 'a'] = node;
   }

   // Get the reference of the node
   TrieNode get(char ch){
    return links[ch - 'a']; 
   }

   // Sets the end of a word
   void setEnd(){
    this.flag = true;
   }

   boolean isEnd(){
    return this.flag;
   }
}

class Trie {

    private TrieNode root;

    public Trie() {
        this.root = new TrieNode();
    }
    
    public void insert(String word) {
        
        TrieNode currNode = root;

        for(int i = 0; i < word.length(); i++){

            char ch = word.charAt(i);

            // Put the new alphabet if not already present
            if(!currNode.containsKey(ch)){
                currNode.put(ch, new TrieNode());
            }

            // Move to the ref of newly created node
            currNode = currNode.get(ch);
        }

        // Set the end of the word
        currNode.setEnd();
    }
    
    public boolean search(String word) {
        
        TrieNode currNode = root;

        for(int i = 0; i < word.length(); i++){

            char ch = word.charAt(i);

            if(!currNode.containsKey(ch)){
                return false;
            }

            currNode = currNode.get(ch);
        }

        // Returns true if the end node's flag is true, otherwise false
        return currNode.isEnd();
    }
    
    public boolean startsWith(String prefix) {
        
        TrieNode currNode = root;

        for(int i = 0; i < prefix.length(); i++){

            char ch = prefix.charAt(i);

            if(!currNode.containsKey(ch)){
                return false;
            }

            currNode = currNode.get(ch);
        }

        return true;
    }
}
