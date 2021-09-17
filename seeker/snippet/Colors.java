//date: 2021-09-17T17:12:35Z
//url: https://api.github.com/gists/8c705e1fbfce386f742b464a814c6bc9
//owner: https://api.github.com/users/rkharsan

class Box {
	int len, width, height;

	Box(int a, int b, int c) {
		len = a;
		width = b;
		height = c;
	}

	public double Box_vol() {
		return len * width * height;
	}
}

class BoxWeight extends Box {
	double density;

	BoxWeight(int l, int w, int h, double d) {
		super(l, w, h);
		density = d;
	}

	public double Box_wt() {
		double vol = super.Box_vol();
		return vol * density;
	}
}

class ColorBox extends BoxWeight {
	double cost;
	String color;

	ColorBox(int l, int w, int h, double d, double cost, String color) {
		super(l, w, h, d);
		this.cost = cost;
		this.color = color;
	}

	public double Color_cost() {
		return cost * Box_vol();
	}
}

public class Colors {
	public static void main(String[] args) {
		ColorBox cb1 = new ColorBox(3, 4, 5, 2.4, 8.5, "Red");

		if (cb1.color.equals("White")) {
			System.out.println(cb1.Color_cost());
		} else if (cb1.color.equals("Red")) {
			System.out.println(cb1.Color_cost() + 50);
		} else {
			System.out.println(cb1.Color_cost() + 100);
		}
	}
}
