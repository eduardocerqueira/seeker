//date: 2022-07-18T17:11:56Z
//url: https://api.github.com/gists/2ec91499560a5782cd33a55853f16a8e
//owner: https://api.github.com/users/BetterProgramming

n=inBoard.length;
// Detect type of game: 4x4, 9x9 (traditional Sudoku) 
// or 25x25 (Nozeku)
if (n == 4){
  validChars = "1234";
  charsMask = 0xF;
} else {if (n == 9){
  validChars = "123456789";
  charsMask = 0x1FF;
} else {
  validChars = "ABCDEFGHIJKLMNOPQRSTUVWXY";
  charsMask = 0x1FFFFFF;
}