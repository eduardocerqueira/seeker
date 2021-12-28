//date: 2021-12-28T16:56:49Z
//url: https://api.github.com/gists/4f989ad3491cf374a58ae9b0510b30ca
//owner: https://api.github.com/users/Damini-22

class Solution
{
    //Function to find if the given edge is a bridge in graph.
    static int isBridge(int V, ArrayList<ArrayList<Integer>> adj,int c,int d){
        // code here
        
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        for(int i=0;i<V;i++)
        {
            graph.add(new ArrayList<>());
        }
        for(int i=0; i<V; i++){
            for(int  nbr : adj.get(i) )
            {
                if(!((i==c && nbr==d) || (i==d && nbr==c)))
                    graph.get(i).add(nbr);
            }
        }
        
        
        boolean vis1[]=new boolean[V];
        boolean vis2[]=new boolean[V];
       int c1=0,c2=0; 
       for(int i=0;i<V;i++)
       {
           if(!vis1[i])
           {
                dfs(adj,i,vis1);
                c1++;
           }
       }
       
      for(int i=0;i<V;i++)
      {
          if(!vis2[i])
          {
              dfs(graph,i,vis2);
              c2++;
          }
      }
      //System.out.println(c1+" "+c2);
       return c1==c2?0:1;
    }
   public static void dfs(ArrayList<ArrayList<Integer>> adj,int src,boolean vis[])
   {
        vis[src]=true;
        for(int nbr:adj.get(src))
        {
            if(!vis[nbr])
            {
                dfs(adj,nbr,vis);
            }
        }
        return;
    }
}