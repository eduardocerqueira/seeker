//date: 2023-05-11T17:00:55Z
//url: https://api.github.com/gists/1196e9d235c4fc35c0964f56fde30a90
//owner: https://api.github.com/users/DonMassa84


import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GarageGUI {
    private JFrame frame;
    private JTextField licensePlateField;
    private JTextArea outputArea;
    private Garage garage;

    public GarageGUI() {
        garage = new Garage(3, 5); // 3 Etagen, 5 Parkplätze pro Etage
        initComponents();
    }

    private void initComponents() {
        frame = new JFrame("Parkhaus-Simulator");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(600, 400);

        Container contentPane = frame.getContentPane();
        contentPane.setLayout(new BorderLayout());

        JPanel inputPanel = new JPanel();
        inputPanel.setLayout(new FlowLayout());
        contentPane.add(inputPanel, BorderLayout.NORTH);

        licensePlateField = new JTextField(10);
        inputPanel.add(licensePlateField);

        JButton parkCarButton = new JButton("Auto parken");
        inputPanel.add(parkCarButton);
        parkCarButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                parkCar();
            }
        });

        JButton parkMotorcycleButton = new JButton("Motorrad parken");
        inputPanel.add(parkMotorcycleButton);
        parkMotorcycleButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                parkMotorcycle();
            }
        });

        JButton removeVehicleButton = new JButton("Fahrzeug entfernen");
        inputPanel.add(removeVehicleButton);
        removeVehicleButton.addActionListener(new ActionListener() {
            public void actionPerformed(ActionEvent e) {
                removeVehicle();
            }
        });

        outputArea = new JTextArea();
        JScrollPane scrollPane = new JScrollPane(outputArea);
        contentPane.add(scrollPane, BorderLayout.CENTER);

        frame.setVisible(true);
    }

    private void parkCar() {
        String licensePlate = licensePlateField.getText();
        Car car = new Car(licensePlate);
        boolean success = garage.parkVehicle(car);
        if (success) {
            outputArea.append("Auto (" + licensePlate + ") erfolgreich geparkt.\n");
        } else {
            outputArea.append("Auto (" + licensePlate + ") konnte nicht geparkt werden. Kein freier Parkplatz.\n");
        }
    }

    private void parkMotorcycle() {
        String licensePlate = licensePlateField.getText();
        Motorcycle motorcycle = new Motorcycle(licensePlate);
        boolean success = garage.parkVehicle(motorcycle);
        if (success) {
            outputArea.append("Motorrad (" + licensePlate + ") erfolgreich geparkt.\n");
        } else {
            outputArea.append("Motorrad (" + licensePlate + ") konnte nicht geparkt werden. Kein freier Parkplatz.\n");
        }
    }

    private void removeVehicle() {
        String licensePlate = licensePlateField.getText();
        boolean success = garage.removeVehicle(licensePlate);
        if (success) {
            outputArea.append("Fahrzeug (" + licensePlate + ") erfolgreich entfernt.\n");
        } else {
            outputArea.append("Fahrzeug (" + licensePlate + ") konnte nicht gefunden und entfernt werden.\n");
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                new GarageGUI();
            }
        });
    }
}