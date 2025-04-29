//date: 2025-04-29T16:45:51Z
//url: https://api.github.com/gists/53863bd4e460baed0b99193e3b8674ad
//owner: https://api.github.com/users/kishida

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.event.ActionListener;
import java.net.http.HttpClient;
import java.time.LocalTime;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

public class ToolSample {
    static class WeatherService {
        String weather = "雨";
        int temperature = 15;
        @Tool("return weather of the location")
        String getWeather(String location) {
            System.out.println("getWeather called");
            return weather;
        }
        
        @Tool("return temperature in celsius of the location")
        int getTemperature(String location) {
            System.out.println("getTemperature called");
            return temperature;
        }
        @Tool("return current time") 
        String getCurrentTime() {
            System.out.println("getCurrentTime called");
            return LocalTime.now().toString();
        }
    }
    
    interface Assistant {
        TokenStream chat(String userMessage);
    }
    
    //static String MODEL = "gemma-3-12b-it";
    static String MODEL = "qwen3-1.7b";
    //static String MODEL = "qwen3-4b";
    public static void main(String[] args) {
        var model = OpenAiStreamingChatModel.builder()
                .baseUrl("http://localhost:1234/v1")
                .modelName(MODEL)
                .httpClientBuilder(JdkHttpClient.builder().httpClientBuilder(
                        HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)))
                .build();        
        
        var service = new WeatherService();
        var assistant = AiServices.builder(Assistant.class)
                .streamingChatLanguageModel(model)
                .systemMessageProvider(memId -> """
                        /nothink
                        あなたは天気や気温、現在時刻を返すことができるエージェントです。
                        """)
                .tools(service)
                .build();
        JFrame f = new JFrame("天気アプリ");
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setSize(400, 300);
        var panel = new JPanel(new GridLayout(2, 1));
        var top = new JPanel(new FlowLayout());
        var bottom = new JPanel(new FlowLayout());
        var text = new JTextField(25);
        var button = new JButton("投稿");
        top.add(text);
        top.add(button);
        panel.add(top);
        var combo = new JComboBox<String>(new String[]{"晴れ", "曇り", "雨", "雪"});
        bottom.add(combo);
        var slider = new JSlider(-10, 50, 20);
        bottom.add(slider);
        panel.add(bottom);
        f.add(BorderLayout.NORTH, panel);
        var area = new JTextArea();
        f.add(new JScrollPane(area));
        f.setVisible(true);
        ActionListener event = ae -> {
            service.temperature = slider.getValue();
            service.weather = combo.getSelectedItem().toString();
            var prompt = text.getText();
            Thread.ofVirtual().start(() -> {
                SwingUtilities.invokeLater(() -> {
                    area.append("> %s\n".formatted(prompt));
                    text.setText(prompt);
                });
                
                var response = assistant.chat(text.getText());
                response
                        .onPartialResponse(str -> SwingUtilities.invokeLater(() -> area.append(str)))
                        .ignoreErrors()
                        .onToolExecuted(te -> {
                            SwingUtilities.invokeLater(() -> area.append(
                                    "%s invoked with %s\n".formatted(te.request().name(), te.request().arguments())));
                        })
                        .onCompleteResponse(resp -> SwingUtilities.invokeLater(() -> area.append("\n")))
                        .start();
            });
        };
        button.addActionListener(event);
        text.addActionListener(event);
    }
}
