//date: 2025-06-09T16:44:31Z
//url: https://api.github.com/gists/bc7cec2d036c906111a5be93f1159870
//owner: https://api.github.com/users/kishida

package com.mycompany.langsample;

import dev.langchain4j.http.client.jdk.JdkHttpClient;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.openai.OpenAiStreamingChatModel;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.service.TokenStream;
import java.awt.BorderLayout;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.net.http.HttpClient;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTabbedPane;
import javax.swing.JTextArea;
import javax.swing.SwingUtilities;

public class DevstralCodingAgent {
    static JFrame frame;
    static JTabbedPane codeArea;
    static JTextArea logArea;
    
    record SrcFile(String path, StringBuilder code, JScrollPane area) {       
    }
    static Map<String, SrcFile> files = new HashMap<>();
    
    enum State {PLAIN, IN_XML, IN_CODE};
    
    interface Handler {
        void plain(char ch);
        void sourceStarted();
        void filename(String filename);
        void code(String code);
        void sourceEnded();
    }
    
    static class ResponseReader {
        Handler handler;
        State state;
        StringBuilder buf;
        boolean formatted = false;
        
        static final char RET = 0x0a;

        public ResponseReader(Handler handler) {
            this.handler = handler;
            state = State.PLAIN;
            buf = new StringBuilder();
        }
        
        void consume(char ch) {
            buf.append(ch);
            if (state == State.PLAIN) {
                handler.plain(ch);
            }
            if (ch != RET) {
                return;
            }
            String str = buf.toString();
            buf.setLength(0);
            switch (state) {
                case PLAIN -> {
                    if (str.startsWith("```")) {
                        formatted = !formatted;
                    } else if (str.trim().equals("<source>")) {
                        state = State.IN_XML;
                        handler.sourceStarted();
                    }
                }
                case IN_XML -> {
                    if (str.startsWith("<filename>")) {
                        String name = str.substring("<filename>".length(), 
                                str.length() - "</filename>\n".length());
                        handler.filename(name);
                    } else if (str.trim().equals("<code>")) {
                        state = State.IN_CODE;
                    } else if (str.trim().equals("</source>")) {
                        state = State.PLAIN;
                        handler.sourceEnded();
                    }else {
                        System.out.println(str);
                    }
                }
                case IN_CODE -> {
                    if (str.trim().equals("</code>")) {
                        state = State.IN_XML;
                    } else {
                        String code = str.replaceAll("&lt;", "<").replaceAll("&gt;", ">"
                                .replaceAll("&quot;", "\"").replaceAll("&amp;", "&"));
                        handler.code(code);
                    }
                }
            }            
        }
    }
    
    static Handler handler = new Handler() {
        @Override
        public void plain(char ch) {
            logArea.append(ch + "");
        }

        
        @Override
        public void sourceStarted() {
        }

        JTextArea selected;
        SrcFile src;
        
        @Override
        public void filename(String filename) {
            if (files.containsKey(filename)) {
                src = files.get(filename);
                src.code.setLength(0);
                SwingUtilities.invokeLater(() -> {
                    codeArea.setSelectedComponent(src.area());
                    selected = (JTextArea)((JScrollPane)src.area).getViewport().getView();
                    selected.setText("");
                });
            } else {
                selected = new JTextArea();
                src = new SrcFile(filename, new StringBuilder(), new JScrollPane(selected));
                files.put(filename, src);
                Path p = Path.of(filename);
                SwingUtilities.invokeLater(() -> {
                    codeArea.add(p.getFileName().toString(), src.area());
                });
            }
            SwingUtilities.invokeLater(() -> codeArea.setSelectedComponent(src.area()));
        }

        @Override
        public void code(String code) {
            src.code.append(code);
            selected.append(code);
            selected.setCaretPosition(selected.getDocument().getLength());
        }

        @Override
        public void sourceEnded() {
            logArea.append("saved " + src.path + "\n");
            try {
                var p = Path.of("temp", src.path);
                Files.createDirectories(p.getParent());
                Files.writeString(p, src.code.toString());
            } catch(IOException ex) {
                logArea.append("save error " + ex.getLocalizedMessage());
                ex.printStackTrace();
            }
            
        }
    };
    
    interface Assistant {
        TokenStream chat(String userMessage);
    }
    
    static String systemPrompt = """
you are coding agent.
generate source codes following user instruction.
you must generate whole code to run the user intended system including configuration and build script.
you must make readme.md file including the feature, file structure, how to build, how to run.

all generated source codes including readme.md must be in the xml format below.
code tag start and end must be separated line.
<source>
<filename>path of code</filename>
<code>
the code that must use entity reference. not use use CDATA tag.
</code>
</source>
        """;
    // cdata instruction is required for gemma3
    
    public static void main(String[] args) {
        String modelname = 
                //"devstral-small-2505";
                "devstral-small-2505@q3_k_s";
        var model = OpenAiStreamingChatModel.builder()
                .baseUrl("http://localhost:1234/v1")
                .modelName(modelname)
                .httpClientBuilder(JdkHttpClient.builder().httpClientBuilder(
                        HttpClient.newBuilder().version(HttpClient.Version.HTTP_1_1)))
                .build();
        var assistant = AiServices.builder(Assistant.class)
                .streamingChatLanguageModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(30))
                .systemMessageProvider(memId -> systemPrompt)
                .build();
        
        frame = new JFrame("雑コーディングエージェント with " + modelname);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(800, 600);
        var panel = new JPanel();
        var promptField = new JTextArea(5, 50);
        promptField.setText("""
                spring bootでtodo管理アプリを作って。追加、削除、完了が行える。
                thymeleafでのテンプレートとコントローラ、リポジトリ、エンティティも。
                DBにはSpring data JPAとHibernateをつかってH2にアクセスして。エンティティには@Idを付けて。
                @Entityではテーブル名を大文字で指定。
                lombokは禁止。
                初期化用のSQLと、あとはmavenのpomもつくって。
                            """);
        var button = new JButton("生成");
        panel.add(promptField);
        panel.add(button);
        frame.add(BorderLayout.NORTH, panel);
        logArea = new JTextArea(7, 50);
        logArea.setLineWrap(true);
        frame.add(BorderLayout.SOUTH, 
                new JScrollPane(logArea, JScrollPane.VERTICAL_SCROLLBAR_ALWAYS, 
                                         JScrollPane.HORIZONTAL_SCROLLBAR_NEVER));
        
        codeArea = new JTabbedPane();
        frame.add(codeArea);
        frame.setVisible(true);
        
        ResponseReader resRed = new ResponseReader(handler);
        
        ActionListener event = ae -> {
            Thread.ofPlatform().start(() -> {
                var prompt = promptField.getText();
                if (prompt.trim().isBlank()) {
                    return;
                }
                SwingUtilities.invokeLater(() -> logArea.append("> " + prompt + "\n"));
                var response = assistant.chat(prompt);
                response.onPartialResponse(str -> {
                            str.chars().forEach(ch -> resRed.consume((char)ch));
                        })
                        .ignoreErrors()
                        .onCompleteResponse(resp -> SwingUtilities.invokeLater(() -> logArea.append("\n")))
                        .start();
            });
        };
        button.addActionListener(event);
    }
        
}
