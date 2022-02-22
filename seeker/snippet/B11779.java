//date: 2022-02-22T16:56:26Z
//url: https://api.github.com/gists/d8f1d643c3097a01fb883b47aab1a74b
//owner: https://api.github.com/users/shininghyunho

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.*;

class B11779_SOL{
    private static int n,m,from,to;
    private static final int INF=987654321;
    private static List<List<Node>> graph=new ArrayList<>();
    private static int[] dist,route;
    class Node{
        // a->b , d
        int p,d;

        public Node(int p, int d) {
            this.p = p;
            this.d = d;
        }
    }
    public void input() throws IOException{
        BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
        StringTokenizer stk;
        n=Integer.parseInt(br.readLine());
        m=Integer.parseInt(br.readLine());
        for(int i=0;i<n+1;i++) graph.add(new ArrayList<>());
        for(int i=0;i<m;i++){
            stk=new StringTokenizer(br.readLine());
            int a=Integer.parseInt(stk.nextToken());
            int b=Integer.parseInt(stk.nextToken());
            int d=Integer.parseInt(stk.nextToken());
            graph.get(a).add(new Node(b,d));
        }
        stk=new StringTokenizer(br.readLine());
        from=Integer.parseInt(stk.nextToken());
        to=Integer.parseInt(stk.nextToken());
    }
    public void solve(){
        dist=new int[n+1];
        route=new int[n+1];
        dijkstra(from,dist,graph);
        print();
    }
    private void dijkstra(int start,int[] dist,List<List<Node>> graph){
        for(int i=0;i<n+1;i++) dist[i]=INF;
        dist[start]=0;
        route[start]=-1;
        PriorityQueue<Node> pq=new PriorityQueue<>((a,b)->{return a.d-b.d;});
        pq.add(new Node(start,0));
        while(!pq.isEmpty()){
            Node now=pq.poll();
            if(dist[now.p]<now.d) continue;
            for(Node next:graph.get(now.p)){
                int newDist=dist[now.p]+next.d;
                if(newDist<dist[next.p]){
                    dist[next.p]=newDist;
                    route[next.p]=now.p;
                    pq.add(new Node(next.p,newDist));
                }
            }
        }
    }
    private void print(){
        Stack<Integer> st=new Stack<>();
        int now=to;
        while(now!=-1){
            st.add(now);
            now=route[now];
        }
        StringBuilder sb=new StringBuilder();
        sb.append(dist[to]).append("\n");
        sb.append(st.size()).append("\n");
        while(!st.isEmpty()) sb.append(st.pop()).append(" ");
        System.out.println(sb);
    }

}
public class Main {
    public static void main(String[] args) throws IOException{
        B11779_SOL sol=new B11779_SOL();
        sol.input();
        sol.solve();
    }
}