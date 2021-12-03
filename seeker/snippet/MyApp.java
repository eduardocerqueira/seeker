//date: 2021-12-03T17:08:10Z
//url: https://api.github.com/gists/ea1a3726501a785c960c1c6e355b78b5
//owner: https://api.github.com/users/mcrisc

package example;

import java.awt.BorderLayout;
import java.awt.CardLayout;
import java.awt.Color;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.util.Set;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

public class MyApp {

	private static final String TAG_ENTRADA = "entrada";
	private static final String TAG_SAIDA = "saida";
	private Set<String> estacionados = Set.of("ABC-1234", "DEF-5678");
	
	private JPanel cards;
	private JTextField txtPlaca;

	
	private void createAndShowGUI() {
		JFrame frame = new JFrame("Exemplo");
		JPanel panel = buildMainPanel();
		frame.add(panel);
		frame.setSize(400, 300);
		frame.setLocationRelativeTo(null);
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setVisible(true);		
	}

	private JPanel buildMainPanel() {
		JPanel mainPanel = new JPanel();
		mainPanel.setLayout(new BorderLayout());
		
		JPanel topPanel = new JPanel(new FlowLayout());
		JLabel lblPlaca = new JLabel("Placa: ");
		txtPlaca = new JTextField(8);
		txtPlaca.setText("ABC-1234");
		JButton btnPesquisar = new JButton("Pesquisar");
		btnPesquisar.addActionListener(this::onPesquisar);
		
		topPanel.add(lblPlaca);
		topPanel.add(txtPlaca);
		topPanel.add(btnPesquisar);
		mainPanel.add(topPanel, BorderLayout.NORTH);
				
		
		cards = new JPanel(new CardLayout());
		mainPanel.add(cards, BorderLayout.CENTER);
		
		JPanel entradaPanel = buildPanelEntrada();
		JPanel saidaPanel = buildPanelSaida();

		cards.add(entradaPanel);
		cards.add(saidaPanel);
		CardLayout layout = (CardLayout)cards.getLayout();
		layout.addLayoutComponent(entradaPanel, TAG_ENTRADA);
		layout.addLayoutComponent(saidaPanel, TAG_SAIDA);
		
		return mainPanel;
	}

	private JPanel buildPanelSaida() {
		JPanel panel = new JPanel();
		panel.setBackground(Color.YELLOW);
		JLabel lblSaida = new JLabel("Saída");
		panel.add(lblSaida);
		
		return panel;
	}

	private JPanel buildPanelEntrada() {
		JPanel panel = new JPanel();
		JLabel lblEntrada = new JLabel("Entrada");
		panel.add(lblEntrada);
		
		return panel;
	}
	
	private void onPesquisar(ActionEvent evt) {
		final CardLayout layout = (CardLayout)cards.getLayout();
		
		String placa = txtPlaca.getText();		
		if (isSaida(placa)) {
			layout.show(cards, TAG_SAIDA);
		} else {
			layout.show(cards, TAG_ENTRADA);
		}
	}

	private boolean isSaida(String placa) {
		return estacionados.contains(placa.trim());
	}
	
	private void execute() {
		SwingUtilities.invokeLater(() -> {createAndShowGUI();});
	}
	
	public static void main(String[] args) {
		new MyApp().execute();
	}
	
}
