//date: 2022-07-04T03:31:18Z
//url: https://api.github.com/gists/e40e8614eb0afe958d72378ff6563f14
//owner: https://api.github.com/users/zulffaza

package com.faza.example.apachecommonsnetftpintegration.ftp;

import org.apache.commons.net.ProtocolCommandEvent;
import org.apache.commons.net.ProtocolCommandListener;
import org.apache.commons.net.ftp.FTPClient;
import org.springframework.stereotype.Service;

@Service
public class FtpService {

  public FTPClient loginFtp() throws Exception {
    FTPClient ftpClient = new FTPClient();
    ftpClient.addProtocolCommandListener(new ProtocolCommandListener() {
      @Override
      public void protocolCommandSent(ProtocolCommandEvent protocolCommandEvent) {
        System.out.printf("[%s][%d] Command sent : [%s]-%s", Thread.currentThread().getName(),
            System.currentTimeMillis(), protocolCommandEvent.getCommand(),
            protocolCommandEvent.getMessage());
      }

      @Override
      public void protocolReplyReceived(ProtocolCommandEvent protocolCommandEvent) {
        System.out.printf("[%s][%d] Reply received : %s", Thread.currentThread().getName(),
            System.currentTimeMillis(), protocolCommandEvent.getMessage());
      }
    });
    ftpClient.connect("localhost", 2121);
    ftpClient.login("admin", "admin");
    return ftpClient;
  }
}