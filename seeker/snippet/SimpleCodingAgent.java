//date: 2025-04-24T17:09:49Z
//url: https://api.github.com/gists/ee3949c412cfd60fe7c6493af2254562
//owner: https://api.github.com/users/kishida

package com.mycompany.langsample;

import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import java.awt.BorderLayout;
import java.awt.GridLayout;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;

public class SimpleCodingAgent {
    static JFrame frame;
    static JTextArea codeArea;
    static JLabel fileNameLabel;
    static JTextArea logArea;
    static class Tools {
        @Tool("save the code into the file with givin name. parameter: filename - file name to save; code - generated Java source code") 
        void saveCode(String filename, String code) {
            SwingUtilities.invokeLater(() -> {
                codeArea.setText(code);
                fileNameLabel.setText(filename);
                logArea.append("saved " + filename + "\n");
                try {
                    Files.writeString(Path.of("temp", filename), code);
                } catch(IOException ex) {
                    throw new UncheckedIOException(ex);
                }
            });
        }
        @Tool("run the java code saved in the file named the filename. parameter: filenmae - file name that code is saved to execute; return: result")
        String executeCode(String filename) {
            logArea.append("execute " + filename + "\n");
            var str = compileAndRun(Path.of("temp", filename));
            if (!str.isBlank()) logArea.append(str + "\n");
            return str;
        }
    }
    interface Assistant {
        TokenStream chat(String userMessage);
    }
    
    static String systemPrompt = """
        あなたはJavaコードの生成を行うコーディングエージェントです。
        ユーザーの指示に従った完全なコードを生成して、適切なファイル名でsaveCodeで保存を行い、  
        executeCodeでファイル名を指定して実行してください。
        実行がうまくいかなかったらエラーや実行失敗が返ります。コードを修正してやりなおしてください。
        """;
    
    public static void main(String[] args) {
        var model = OpenAiStreamingChatModel.builder()
                .baseUrl("http://localhost:1234/v1")
                .modelName("gemma-3-12b-it-qat-japanese-imatrix")
                .httpClientBuilder(JdkHttpClient.builder().httpClientBuilder(
                        HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)))
                .build();
        var tools = new Tools();
        var assistant = AiServices.builder(Assistant.class)
                .streamingChatLanguageModel(model)
                .systemMessageProvider(memId -> systemPrompt)
                .tools(tools)
                .build();
        
        frame = new JFrame("雑コーディングエージェント");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        var panel = new JPanel(new GridLayout(2, 1));
        var panelT = new JPanel();
        var panelB = new JPanel();
        panel.add(panelT);
        panel.add(panelB);
        var promptField = new JTextField(50);
        promptField.setText("JTableのサンプルを作って。ボタンを押すとなにか編集して。");
        var button = new JButton("生成");
        panelT.add(promptField);
        panelT.add(button);
        fileNameLabel = new JLabel();
        panelB.add(fileNameLabel);
        frame.add(BorderLayout.NORTH, panel);
        logArea = new JTextArea(5, 50);
        frame.add(BorderLayout.SOUTH, logArea);
        
        codeArea = new JTextArea();
        frame.add(new JScrollPane(codeArea));
        frame.setVisible(true);
        
        ActionListener event = ae -> {
            var prompt = promptField.getText();
            if (prompt.trim().isBlank()) {
                return;
            }
            logArea.append("> " + prompt + "\n");
            var response = assistant.chat(prompt);
            response.onPartialResponse(str -> SwingUtilities.invokeLater(() -> logArea.append(str)))
                    .ignoreErrors()
                    .onCompleteResponse(resp -> SwingUtilities.invokeLater(() -> logArea.append("\n")))
                    .start();
        };
        button.addActionListener(event);
    }

   /**
     * Javaソースファイルをコンパイルして実行する(引数の型以外はChatGPT出力そのまま)
     * @param javaFile ソースファイルへのパス（例: src/generated/Hello.java）
     * @return コンパイル・実行の結果（標準出力とエラー出力）
     */
    public static String compileAndRun(Path javaFile) {
        try {
            if (!Files.exists(javaFile)) {
                return "Java file not found: " + javaFile;
            }

            // ファイル名とクラス名
            String fileName = javaFile.getFileName().toString();
            String className = fileName.substring(0, fileName.lastIndexOf("."));
            Path dir = javaFile.getParent();

            // javacでコンパイル
            Process compile = new ProcessBuilder("javac", fileName)
                    .directory(dir.toFile())
                    .redirectErrorStream(true)
                    .start();

            String compileOutput = new String(compile.getInputStream().readAllBytes());
            int compileResult = compile.waitFor();
            if (compileResult != 0) {
                return "コンパイル失敗:\n" + compileOutput;
            }

            // javaで実行
            Process run = new ProcessBuilder("java", className)
                    .directory(dir.toFile())
                    .redirectErrorStream(true)
                    .start();

            String runOutput = new String(run.getInputStream().readAllBytes());
            int runResult = run.waitFor();

            return (runResult == 0 ? "実行成功:\n" : "実行失敗:\n") + runOutput;

        } catch (Exception e) {
            return "エラー: " + e.getMessage();
        }
    }

    
}
