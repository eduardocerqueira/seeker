//date: 2023-11-09T17:08:38Z
//url: https://api.github.com/gists/a414e5bf0a9e19d3eea33d4856bd757b
//owner: https://api.github.com/users/adityadixit07

public class MatrixSnakePattern
{
    static ArrayList<Integer> snakePattern(int matrix[][])

    {

        // code here 

        ArrayList<Integer> res=new ArrayList<>();

        for(int i=0;i<matrix.length;i++){

            for(int j=0;j<matrix[0].length;j++){

                if(i%2==0){

                    res.add(matrix[i][j]);

                }

                else{

                    res.add(matrix[i][matrix[0].length-j-1]);

                }

            }

        }

        return res;

    }

}