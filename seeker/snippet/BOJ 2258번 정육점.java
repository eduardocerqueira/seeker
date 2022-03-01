//date: 2022-03-01T17:05:07Z
//url: https://api.github.com/gists/4257609c7ada83e103c7928209bdb9b5
//owner: https://api.github.com/users/yeonui-0626

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.StringTokenizer;
 
public class Main {

	static class Fork implements Comparable<Fork> {
		int cost, weight;

		public Fork(int weight, int cost) {
			super();
			this.weight = weight;
			this.cost = cost;
		}

		@Override
		public String toString() {
			return "Fork [" + weight + ", " + cost + "]";
		}

		@Override
		public int compareTo(Fork o) {
			if (this.cost == o.cost)
				return o.weight - this.weight;
			return this.cost - o.cost;
		}

	}

	static int N, M;
	static ArrayList<Fork> fork;

	public static void main(String[] args) throws IOException {
		BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
		StringTokenizer st = new StringTokenizer(br.readLine());

		N = Integer.parseInt(st.nextToken());
		M = Integer.parseInt(st.nextToken());

		fork = new ArrayList<Fork>();

		for (int i = 0; i < N; i++) {
			st = new StringTokenizer(br.readLine(), " ");
			fork.add(new Fork(Integer.parseInt(st.nextToken()), Integer.parseInt(st.nextToken())));
		}
		
		// 정렬 -> 비용오름차순, 비용이 같다면 무게 내림차순
		Collections.sort(fork);
		

		// 더 저렴한 덩어리는 그냥 얻을 수 있으므로 작은 비용부터 시작해서 얻을 수 있는 고기를 누적해서 계산한다.
		int sum = 0, cost = 0;
		int Ans = Integer.MAX_VALUE;
		boolean flag = false;
		for (int i = 0 ; i < N; i++) {
			
			sum += fork.get(i).weight; // 얻을 수 있는 고기 누적 계산

			if (i>0 && fork.get(i).cost > fork.get(i - 1).cost) { // 이전비용보다 크다면
				cost = fork.get(i).cost; 	// 해당 비용만 적용
			} else { 						// 이전과 가격이 같다면
				cost += fork.get(i).cost; 	// 비용도 누적계산
			}

			if (sum >= M) { // 누적 무게가 사야할 무게 M 이상이라면
				Ans = Math.min(Ans, cost); // 현재 비용과 비교하여 더 작은 비용 저장
				flag = true;
			}
		}
		
		if (flag)
			System.out.println(Ans);
		else
			System.out.println(-1);
	}
	
}
