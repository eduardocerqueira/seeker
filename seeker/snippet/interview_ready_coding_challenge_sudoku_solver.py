#date: 2025-01-03T16:47:02Z
#url: https://api.github.com/gists/863ae0dd5fcc20b51ba7c5443dee64c5
#owner: https://api.github.com/users/mariano22

class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        candidates = [[ [] for _ in range(9)] for _ in range(9)]

        def neighboors(i,j):
            same_row = [(i,x) for x in range(9)]
            same_column = [(x,j) for x in range(9)]
            i_upperleft_box = (i//3)*3
            j_upperleft_box = (j//3)*3
            same_box = [(i_upperleft_box+di, j_upperleft_box+dj) for di in range(3) for dj in range(3)]
            result = set(same_row + same_column + same_box)
            return result

        def update_candidates(i,j):
            candidates_ij = set(str(c) for c in range(1,10))
            for x,y in neighboors(i,j):
                c = board[x][y]
                if c!='.':
                    candidates_ij.discard(c)
            candidates[i][j]=candidates_ij

        def choose_less_candidates():
            x,y = None, None
            already_had = 10
            for i in range(9):
                for j in range(9):
                    if board[i][j]=='.' and len(candidates[i][j]) < already_had:
                        x = i
                        y = j
                        already_had = len(candidates[i][j])
            return x,y

        for i in range(9):
            for j in range(9):
                update_candidates(i,j)

        def backtracking():
            x,y = choose_less_candidates()
            #print(x,y, len(candidates[x][y]))
            if x == None:
                return True
            to_modify = neighboors(x,y)
            for c in candidates[x][y]:
                # TODO: Modify state
                board[x][y] = c
                for i,j in to_modify:
                    update_candidates(i,j)
                if backtracking():
                    return True
                # TODO Un-modify state
                board[x][y] = '.'
                for i,j in to_modify:
                    update_candidates(i,j)
            return False

        backtracking()