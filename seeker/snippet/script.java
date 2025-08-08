//date: 2025-08-08T16:58:18Z
//url: https://api.github.com/gists/db1572ab4a18ff084223e611b74ee0dc
//owner: https://api.github.com/users/AnthonyJusu

public class WeatherApp extends JFrame {

    private JTextField cityInput;
    private JLabel tempLabel, humidityLabel, windLabel, conditionLabel, iconLabel, backgroundLabel;
    private JTextArea historyArea;
    private JComboBox<String> tempUnit;
    private ArrayList<String> searchHistory = new ArrayList<>();
    private String apiKey = "YOUR_API_KEY"; // Replace with your OpenWeatherMap API key

    public WeatherApp() {
        setTitle("Weather Information App");
        setSize(600, 500);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(null);

        // Background image label
        backgroundLabel = new JLabel();
        backgroundLabel.setBounds(0, 0, 600, 500);
        add(backgroundLabel);

        JLabel cityLabel = new JLabel("Enter City:");
        cityLabel.setBounds(20, 20, 100, 25);
        add(cityLabel);

        cityInput = new JTextField();
        cityInput.setBounds(100, 20, 200, 25);
        add(cityInput);

        JButton searchButton = new JButton("Get Weather");
        searchButton.setBounds(320, 20, 120, 25);
        add(searchButton);

        tempUnit = new JComboBox<>(new String[]{"Celsius", "Fahrenheit"});
        tempUnit.setBounds(460, 20, 100, 25);
        add(tempUnit);

        tempLabel = new JLabel("Temperature: --");
        tempLabel.setBounds(20, 70, 300, 25);
        add(tempLabel);

        humidityLabel = new JLabel("Humidity: --");
        humidityLabel.setBounds(20, 110, 300, 25);
        add(humidityLabel);

        windLabel = new JLabel("Wind Speed: --");
        windLabel.setBounds(20, 150, 300, 25);
        add(windLabel);

        conditionLabel = new JLabel("Condition: --");
        conditionLabel.setBounds(20, 190, 300, 25);
        add(conditionLabel);

        iconLabel = new JLabel();
        iconLabel.setBounds(350, 70, 200, 200);
        add(iconLabel);

        JLabel historyLabel = new JLabel("Search History:");
        historyLabel.setBounds(20, 250, 200, 25);
        add(historyLabel);

        historyArea = new JTextArea();
        JScrollPane scrollPane = new JScrollPane(historyArea);
        scrollPane.setBounds(20, 280, 540, 150);
        add(scrollPane);

        // Event handler for search button
        searchButton.addActionListener(e -> fetchWeather());

        updateBackground();
    }

    private void fetchWeather() {
        String city = cityInput.getText().trim();
        if (city.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please enter a city name.");
            return;
        }

        try {
            String unit = tempUnit.getSelectedItem().toString().equals("Celsius") ? "metric" : "imperial";
            String urlString = "https://api.openweathermap.org/data/2.5/weather?q=" + city + "&units=" + unit + "&appid=" + apiKey;

            URL url = new URL(urlString);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");

            BufferedReader br = new BufferedReader(new InputStreamReader(conn.getInputStream()));
            StringBuilder response = new StringBuilder();
            String line;
            while ((line = br.readLine()) != null) {
                response.append(line);
            }
            br.close();

            JSONObject json = new JSONObject(response.toString());
            double temp = json.getJSONObject("main").getDouble("temp");
            int humidity = json.getJSONObject("main").getInt("humidity");
            double wind = json.getJSONObject("wind").getDouble("speed");
            String condition = json.getJSONArray("weather").getJSONObject(0).getString("description");
            String icon = json.getJSONArray("weather").getJSONObject(0).getString("icon");

            tempLabel.setText("Temperature: " + temp + " " + (unit.equals("metric") ? "°C" : "°F"));
            humidityLabel.setText("Humidity: " + humidity + "%");
            windLabel.setText("Wind Speed: " + wind + (unit.equals("metric") ? " m/s" : " mph"));
            conditionLabel.setText("Condition: " + condition);

            // Set weather icon
            ImageIcon weatherIcon = new ImageIcon(new URL("http://openweathermap.org/img/wn/" + icon + "@2x.png"));
            iconLabel.setIcon(weatherIcon);

            // Add to history
            String timestamp = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
            searchHistory.add(city + " - " + temp + "° " + condition + " at " + timestamp);
            historyArea.setText(String.join("\n", searchHistory));

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Error: Unable to fetch weather data.");
        }
    }

    private void updateBackground() {
        int hour = new Date().getHours();
        String bgPath;
        if (hour >= 6 && hour < 18) {
            bgPath = "day_background.jpg";
        } else {
            bgPath = "night_background.jpg";
        }
        backgroundLabel.setIcon(new ImageIcon(bgPath));
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            WeatherApp app = new WeatherApp();
            app.setVisible(true);
        });
    }
}
