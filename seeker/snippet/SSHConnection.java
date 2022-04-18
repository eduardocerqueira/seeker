//date: 2022-04-18T16:57:18Z
//url: https://api.github.com/gists/2e1e76df6b81a9a6448cf2858a53dea8
//owner: https://api.github.com/users/geeksreeni

import com.jcraft.jsch.*;

import java.io.InputStream;

public class SSHConnection {
    public static Session masterSession;
    public static Channel masterChannel;
    public static Session remoteSession;

    public static void main(String[] args) {
        String command1 = "cd /opt/mzeal/cyglass/collector/server/scripts;./collector_deployment_controller.sh --list";
        String command2 = "./collector_deployment_controller.sh --list";// commad to execute on the server
        connect(); // making a connection to the server
        String result = executeCommand(masterChannel, command1); // executing the command
        System.out.println(result);
//        String result2 = executeCommand(masterChannel, command2); // executing the command
//        System.out.println(result2);
        disconnectChannel(masterChannel);
    }

    public static void connect() {
        try {

            JSch jsch = new JSch();
            String user = "hadoopuser"; // your username
            String host = "qasite3.cyglass.com"; // your secure server address
            String password = "nimdaLeaz@305";

            masterSession = jsch.getSession(user, host);
            masterSession.setConfig("PreferredAuthentications", "publickey,keyboard-interactive,password");
            masterSession.setPassword(password);
            masterSession.setConfig("StrictHostKeyChecking", "no");
            masterSession.setTimeout(15000);
            masterSession.connect();
            System.out.println("ssh connected");
            masterChannel = masterSession.openChannel("exec");

        } catch (JSchException e) {
            e.printStackTrace();
        }

    }

    private static String executeCommand(Channel channel,String command) {
        String finalResult = "";
        try {
            String result = null;

            //Channel channel = masterSession.openChannel("exec");
            ((ChannelExec) channel).setCommand(command);

            channel.setInputStream(null);// this method should be called before connect()

            ((ChannelExec) channel).setErrStream(System.err);
            InputStream inputStream = channel.getInputStream();
            channel.connect();
            byte[] byteObject = new byte[10240];
            while (true) {
                while (inputStream.available() > 0) {
                    int readByte = inputStream.read(byteObject, 0, 1024);
                    if (readByte < 0)
                        break;
                    result = new String(byteObject, 0, readByte);
                    finalResult = finalResult + result;
                }
                if (channel.isClosed())
                    break;
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
        return finalResult;

    }

    public static void disconnectChannel(Channel channel){
        channel.disconnect();
        System.out.println("Disconnected channel " + channel.getExitStatus());
    }
}
