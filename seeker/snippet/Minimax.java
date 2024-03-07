//date: 2024-03-07T16:56:38Z
//url: https://api.github.com/gists/9a6d4c7826fc0ea9c3cf12e5228ac4cc
//owner: https://api.github.com/users/nikhil-RGB


public static int[] minMax(TicTacToeMin obj) 
	{
	 ArrayList<Integer> scores=new ArrayList<>(0);
	 ArrayList<TicTacToeMin> arrs=obj.simulate();
	 ArrayList<Integer> moves=obj.emptySpaces();
	 //return a list of cloned boards with current token fitted into all empty spaces.
	 if(arrs.size()==0) 
	 {
		 return new int[] {0,-1};
	 }
	 
	 for(TicTacToeMin board:arrs) 
	 {
		int score=board.score();
		if(score==-2) 
		{
			score=minMax(board)[0];
		}
		scores.add(score);
	 }
	 
	 int optimalScore;
	 if(obj.currentTok.equals("O")) 
	 {
		 optimalScore=Collections.max(scores);
	 }
	 else 
	 {
		 optimalScore=Collections.min(scores);
	 }
	 
	 return new int[] {optimalScore,moves.get(scores.indexOf(optimalScore))};
	 
	}
	
