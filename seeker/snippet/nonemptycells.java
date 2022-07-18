//date: 2022-07-18T17:10:17Z
//url: https://api.github.com/gists/fb6cb04b6e974e89a626108277da0f37
//owner: https://api.github.com/users/BetterProgramming

// Collect stats on non-empty cells in rows/columns/submatrices
// into bit-masks, 1 bit per digit (1-9).
for (int r = 0; r < n; r++) {
  for (int c = 0; c < n; c++){
    int mr, mc;
    if (board[r][c] != emptyChar) {
      int tokenIndex = validChars.indexOf(board[r][c]);
      rowDigitsMask[r] = rowDigitsMask[r] | digitsMask[tokenIndex];
      colDigitsMask[c] = colDigitsMask[c] | digitsMask[tokenIndex];
      mr = r / sqrtN; mc = c / sqrtN;
      submatDigitsMask[mr][mc] = submatDigitsMask[mr][mc] |
                                   digitsMask[tokenIndex]
      rowDigitsCount[r]++; 
      colDigitsCount[c]++; 
      submatDigitsCount[mr][mc]++;
    }
  }
}
// Process only rows with some empty cells.
// Create bit-masks for each empty cell, 
// 1 bit for every valid candidate digit.
emptyCellCount = 0;
for (int r = 0; r < n; r++){
  for (int c = 0; c < n; c++){
    if (board[r][c] == emptyChar) {
      // Process empty cell at r,c
      emptyCellCount++;
      int mr, mc;
      mr = r / sqrtN; mc = c / sqrtN;
      // Complement DigitsMasks to get Mask of valid PossibleDigits
      // Use bitwise AND (&) to find Candidate digits 
      // valid in all 3 'dimensions'
      possibleDigits[r][c] = charsMask &
                             ~submatDigitsMask[mr][mc] &
                             ~rowDigitsMask[r] &
                             ~colDigitsMask[c];
      if (possibleDigits[r][c] == 0) return -1;
      // Count the number of valid Possible digits in cell [r][c]
      possibleDigitsCount[r][c] = 0;
      for (int msk = 0; msk < n; msk++) {
        if ((possibleDigits[r][c] & digitsMask[msk]) != 0) {
          possibleDigitsCount[r][c]++;  // # candidates for cell
          rowCandidatesCount[r][msk]++; // # occ. of cand. in row
          colCandidatesCount[c][msk]++; // # occ. of cand. in col
          submatCandidatesCount[mr][mc][msk]++;
                                        // # occ. of cand. in sbmtrx
        }
      }
    }
  }
}