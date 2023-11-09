//date: 2023-11-09T17:05:18Z
//url: https://api.github.com/gists/7ddc691258136a5014b041165f6705bc
//owner: https://api.github.com/users/adityadixit07

  int columnWithMaxZeros(int arr[][], int N)
    {
        // code here 
        int max=0;
        int col=-1;
        for(int i=0;i<N;i++){
            int count=0;
            for(int j=0;j<N;j++){
                if(arr[j][i]==0){
                    count++;
                }
            }
            if(count>max){
                max=count;
                col=i;
            }
        }
        return col;
    }