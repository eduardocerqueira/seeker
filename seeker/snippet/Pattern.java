//date: 2022-02-08T16:57:40Z
//url: https://api.github.com/gists/8b54339ded605d6e307942b7d753b21c
//owner: https://api.github.com/users/kumarrcpm

package patternPrograms;

public class Pattern {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
Pattern user = new Pattern();
//user.pattern1();
//user.pattern2();
//user.pattern3();
//user.pattern4();
//user.pattern5();
user.pattern6();
user.pattern7();
user.pattern8();

	}
	
public void pattern8()
{
	for(int row=1;row<=4;row++)
	{
		for(int col=1;col<=5;col++)
		{
			System.out.print("*"+ " ");
		}
		System.out.println();
	}
}
public void pattern7()
{
	for(int row=1;row<=3;row++)
	{
		for(int col=5;col>=1;col--)
		{
			System.out.print(col + " ");
		}
		System.out.println();
	}
}
	
public void pattern6()
{
	for(int row=3;row>=1;row--)
	{
		for(int col=1;col<=5;col++)
		{
			System.out.print(row+" ");
		}
		System.out.println();
	}
}
	
public void pattern5()
{
	for(int row=1;row<=3;row++)
	{
	for (int col=1;col<=5;col++)
	{
		System.out.print(row+" ");
	}
	System.out.println();
	}
}
	
public void pattern4()
{
	for(int row=1;row<=3;row++)
	{
	for (int col=1;col<=5;col++)
	{
		System.out.print(col+" ");
	}
	System.out.println();
	}
}
	
public void pattern3()
{
	for(int row=1;row<=3;row++)
	{
	for (int col=1;col<=5;col++)
	{
		System.out.print(1+" ");
	}
	System.out.println();
	}
}
	
public void pattern2()
{
	for(int col=1;col<=5;col++)
	{
		System.out.print(1+" ");
	}
	System.out.println();
	for(int col=1;col<=5;col++);
	{
		System.out.print(1 +" ");
	}
	System.out.println();
	for(int col=1;col<=5;col++)
	{
		System.out.print(1+" ");
	}
	System.out.println();
}
	
public void pattern1()
{
for(int col=1;col<=5;col++)
{
	System.out.print(1+ " ");
}
}
}