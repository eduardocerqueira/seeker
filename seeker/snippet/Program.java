//date: 2021-11-03T17:06:53Z
//url: https://api.github.com/gists/8a410829c88379d4cca3c095101810a4
//owner: https://api.github.com/users/paulosergioamorim

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.awt.Color.DARK_GRAY;
import static java.awt.Color.LIGHT_GRAY;

/**
 * Jogo da Forca
 * @author Paulo Sergio
 * @author Nycolas Monjardim
 */
public class Program extends JFrame {
    // Constantes
    private static final String[] PALAVRAS = {
            "Instituto Federal do Espírito Santo",
            "Programação Orientada a Objetos",
            "Banco de Dados",
            "Análise e Projeto de Sistemas",
            "Java",
            "Python",
            "Modelo de Entidade e Relacionamento",
            "Modelo Relacional Normalizado",
            "Diagrama de Classes",
            "BrModelo",
            "NetBeans",
            "Framework"
    };

    // Elementos da tela
    private final JLabel Erros;
    private final JLabel Lances;
    private final JTextField Palavra;
    private final JTextField Lance;
    private final JButton Jogar;
    private final JButton NovaTentativa;

    // Atributos
    private String palavra;
    private StringBuilder escondida;
    private String lances;
    private List<String> erros;

    /**
     * Construtor da classe Program
     */
    public Program(String title) {
        super(title);
        super.setDefaultCloseOperation(EXIT_ON_CLOSE);
        super.setResizable(false);
        super.setSize(400,300);

        this.add(Erros);
        this.add(Lances);
        this.add(Palavra);
        this.add(Lance);
        this.add(Jogar);
        this.add(NovaTentativa);

        this.iniciar();
    }
    
    {
        JPanel panel = new JPanel();
        panel.setLayout(null);
        panel.setBackground(DARK_GRAY);
        this.setContentPane(panel);

        Erros = new JLabel("Erros: 0 de 6");
        Erros.setForeground(LIGHT_GRAY);
        Erros.setBounds(10,0,100,30);

        Lances = new JLabel("Lances:");
        Lances.setForeground(LIGHT_GRAY);
        Lances.setBounds(10,30,200,30);

        Palavra = new JTextField();
        Palavra.setFont(new Font("Mono",Font.BOLD,14));
        Palavra.setEnabled(false);
        Palavra.setBounds(10,60,300,30);

        Lance = new JTextField();
        Lance.setToolTipText("Somente o primeiro caractere inserido será considerado");
        Lance.setBounds(60,100,30,30);

        Jogar = new JButton("Jogar");
        Jogar.setFocusPainted(false);
        Jogar.setBounds(10,140,75,30);
        Jogar.addActionListener(e -> this.jogar());

        NovaTentativa = new JButton("Nova Tentativa");
        NovaTentativa.setFocusPainted(false);
        NovaTentativa.setBounds(90,140,120,30);
        NovaTentativa.addActionListener(e -> this.iniciar());

        JLabel lanceLabel = new JLabel("Lance:");
        lanceLabel.setForeground(LIGHT_GRAY);
        lanceLabel.setBounds(10,100,50,30);
        this.add(lanceLabel);
    } // Inicia os componentes da tela e configura os listeners

    /**
     * Inicia o jogo
     */
    public void iniciar() {
        erros = new ArrayList<>();
        lances = "";
        palavra = getPalavra();
        escondida = new StringBuilder(palavra
                .replaceAll("\\p{javaLetter}", "_")
                .replaceAll("\\s", "-"));
        Palavra.setText(escondida.toString());
        this.atualizar();
    }

    /**
     * Percorre a palavra e verifica se o caractere
     * inserido é igual ao da palavra, se for, substitui
     * o '_' da palavra escondida pelo caractere, se não,
     * adiciona o caractere à lista de erros.
     */
    public void jogar() {
        String lance = Lance.getText().toLowerCase();
        if (lance.isEmpty() || lances.contains(lance)) {
            this.atualizar();
            return;
        }
        lance = lance.substring(0, 1);
        if (palavra.toLowerCase().contains(lance)) {
            for (int i = 0; i < palavra.length(); i++) {
                char c = palavra.toLowerCase().charAt(i);
                if (c == lance.charAt(0)) {
                    escondida.setCharAt(i,lance.charAt(0));
                    if (!lances.contains(lance))
                        lances += lance + " ";
                }
            }
        } else if (!erros.contains(lance)) erros.add(lance);
        this.atualizar();
    }

    /**
     * Atualiza os componentes da tela e verifica se o jogo acabou
     */
    public void atualizar() {
        Lance.setText(null);
        Erros.setText("Erros: " + erros.size() + " de 6");
        Lances.setText("Lances: " + lances);
        Palavra.setText(escondida.toString());

        if (erros.size() == 6) {
            JOptionPane.showMessageDialog(this,"Você perdeu!");
            this.iniciar();
        }
        if (!escondida.toString().contains("_")) {
            JOptionPane.showMessageDialog(this,"Você venceu!");
            this.iniciar();
        }
    }

    /**
     * Retorna uma palavra aleatória da lista de palavras
     * @return String
     */
    public String getPalavra() {
        Random random = new Random();
        int i = random.nextInt(PALAVRAS.length - 1);
        return PALAVRAS[i];
    }

    /**
     * Método main
     */
    public static void main(String[] args) {
        JFrame frame = new Program("Jogo da Forca");
        frame.setVisible(true);
    }
}