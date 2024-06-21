//date: 2024-06-21T17:13:11Z
//url: https://api.github.com/gists/5e5e2f8dec0652e2adc115ae91c70dac
//owner: https://api.github.com/users/prabhugopal

import com.sun.tools.attach.VirtualMachine;

public class AttachAgent {

    public static void main(String[] args) throws Exception {
        String pid = args[0];
        String agentPath = args[1];
        VirtualMachine virtualMachine = VirtualMachine.attach(pid);
        virtualMachine.loadAgentPath(agentPath);
    }
}