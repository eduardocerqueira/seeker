//date: 2021-09-01T17:06:00Z
//url: https://api.github.com/gists/c5cb69fa3854075ecd5e24305b8e0e0e
//owner: https://api.github.com/users/micycle1

public class T2Q extends java.applet.Applet {
	
	public static void main(String[] args) {
		T2Q applet = new T2Q();
		applet.init();
		applet.start();

		// Create a window (JFrame) and make applet the content pane.
		javax.swing.JFrame window = new javax.swing.JFrame("T2Q");
		window.setContentPane(applet);
		window.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		window.pack(); // Arrange the components.
		window.setVisible(true); // Make the window visible.
	 }
  }