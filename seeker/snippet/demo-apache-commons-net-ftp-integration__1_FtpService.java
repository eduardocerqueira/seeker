//date: 2022-07-04T03:09:54Z
//url: https://api.github.com/gists/2fc8416f93074032f2a0d942d244c238
//owner: https://api.github.com/users/zulffaza

package com.faza.example.apachecommonsnetftpintegration.ftp;

import org.apache.commons.net.ftp.FTPClient;
import org.springframework.stereotype.Service;

@Service
public class FtpService {

  public FTPClient loginFtp() throws Exception {
    FTPClient ftpClient = new FTPClient();
    ftpClient.connect("localhost", 2121);
    ftpClient.login("admin", "admin");
    return ftpClient;
  }
}