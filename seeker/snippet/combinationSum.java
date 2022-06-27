//date: 2022-06-27T16:52:33Z
//url: https://api.github.com/gists/f909f0dad76638b9a8d928c6b5662e13
//owner: https://api.github.com/users/archana77archana

class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList();
        int start = 0;
        List<Integer> storeCandidates = new ArrayList();
        
        // We will backtracking our way 
        backTrackSolution(candidates, start, target, storeCandidates, result);
        
        return result;
    }
    
    private void backTrackSolution(int[] candidates, int start, int target, List<Integer> storeCandidates, List<List<Integer>> result) {
        // exit scenario
        if(target < 0)
            return;
        
        // Add rest of the candidates to the result
        if(target == 0)
            result.add(new ArrayList(storeCandidates));
        
        for(int i = start; i < candidates.length; i++) {
            // Classic Backtracking!
            storeCandidates.add(candidates[i]);

            backTrackSolution(candidates, i, target-candidates[i], storeCandidates, result);
            
            storeCandidates.remove(storeCandidates.size() - 1);
        }
    }
}