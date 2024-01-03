//date: 2024-01-03T16:48:53Z
//url: https://api.github.com/gists/185bbe21c97189b72cfa01587539dfdb
//owner: https://api.github.com/users/SA-JackMax

import org.alfresco.jlan.debug.DebugConfigSection;
import org.alfresco.jlan.netbios.server.NetBIOSNameServer;
import org.alfresco.jlan.server.NetworkServer;
import org.alfresco.jlan.server.auth.EnterpriseCifsAuthenticator;
import org.alfresco.jlan.server.auth.acl.DefaultAccessControlManager;
import org.alfresco.jlan.server.config.CoreServerConfigSection;
import org.alfresco.jlan.server.config.GlobalConfigSection;
import org.alfresco.jlan.server.config.InvalidConfigurationException;
import org.alfresco.jlan.server.config.SecurityConfigSection;
import org.alfresco.jlan.server.config.ServerConfiguration;
import org.alfresco.jlan.server.core.DeviceContextException;
import org.alfresco.jlan.server.filesys.DiskDeviceContext;
import org.alfresco.jlan.server.filesys.DiskInterface;
import org.alfresco.jlan.server.filesys.DiskSharedDevice;
import org.alfresco.jlan.server.filesys.FilesystemsConfigSection;
import org.alfresco.jlan.server.filesys.SrvDiskInfo;
import org.alfresco.jlan.smb.server.CIFSConfigSection;
import org.alfresco.jlan.smb.server.SMBServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Profile;
import org.springframework.extensions.config.element.GenericConfigElement;
import org.springframework.stereotype.Service;

import javax.annotation.PostConstruct;

/**
 * Created by dkopel on 3/13/16.
 */
@Service
public class SmbService {

    private BeanFactory beanFactory;

    private UserAuthenticationService userAuthenticationService;

    @Value("${smb.hostname}")
    private String hostname;

    @Value("${smb.domain}")
    private String domain;

    @Value("${smb.share}")
    private String share;

    @Value("${smb.broadcastMask}")
    private String broadcastMask;
    
    private Logger LOGGER = LoggerFactory.getLogger(getClass());

    @Autowired
    public SmbService setUserAuthenticationService(UserAuthenticationService userAuthenticationService) {
        this.userAuthenticationService = userAuthenticationService;
        return this;
    }

    @Autowired
    public SmbService setBeanFactory(BeanFactory beanFactory) {
        this.beanFactory = beanFactory;
        return this;
    }

    @PostConstruct
    public void init()  {
        try {
            ServerConfiguration cfg = new DarkPointServerConfig();

            NetBIOSNameServer netBIOSNameServer = new NetBIOSNameServer(cfg);
            cfg.addServer(netBIOSNameServer);
            SMBServer smbServer = new SMBServer(cfg);
            cfg.addServer(smbServer);

            for (int i = 0; i < cfg.numberOfServers(); i++) {
                NetworkServer server = cfg.getServer(i);
                server.startServer();
            }
        } catch (Exception e) {
            LOGGER.error("Something went terribly wrong with the samba server!");
            e.printStackTrace();
        }

    }

    class ServerConfig extends ServerConfiguration {
        private Logger logger = LoggerFactory.getLogger(getClass());

        private static final int DefaultThreadPoolInit  = 1;
        private static final int DefaultThreadPoolMax   = 1;

        private final int[] DefaultMemoryPoolBufSizes  = { 256, 4096, 16384, 66000 };
        private final int[] DefaultMemoryPoolInitAlloc = {  20,   20,     5,     5 };
        private final int[] DefaultMemoryPoolMaxAlloc  = { 100,   50,    50,    50 };


        public ServerConfig() throws InvalidConfigurationException, DeviceContextException {
            super(hostname);
            setServerName(hostname);

            // DEBUG
            DebugConfigSection debugConfig = new DebugConfigSection(this);
            final GenericConfigElement debugConfigElement = new GenericConfigElement("output");
            final GenericConfigElement logLevelConfigElement = new GenericConfigElement("logLevel");
            logLevelConfigElement.setValue("Info");
            debugConfig.setDebug("org.alfresco.jlan.debug.ConsoleDebug", debugConfigElement);

            // CORE
            CoreServerConfigSection coreConfig = new CoreServerConfigSection(this);
            coreConfig.setMemoryPool( DefaultMemoryPoolBufSizes, DefaultMemoryPoolInitAlloc, DefaultMemoryPoolMaxAlloc);
            coreConfig.setThreadPool(DefaultThreadPoolInit, DefaultThreadPoolMax);
            coreConfig.getThreadPool().setDebug(false);

            // GLOBAL
            GlobalConfigSection globalConfig = new GlobalConfigSection(this);

            // SECURITY
            final SecurityConfigSection secConfig = new SecurityConfigSection(this);
            secConfig.setUsersInterface(
                "connector.smb.UsersInterface",
                new GenericConfigElement("")
            );
            this.addConfigSection(secConfig);

            DefaultAccessControlManager accessControlManager = new DefaultAccessControlManager();
            accessControlManager.setDebug(false);

            // SHARES
            FilesystemsConfigSection filesysConfig = new FilesystemsConfigSection(this);
            DiskInterface diskInterface = beanFactory.getBean(InMemoryFileDriver.class);
            final GenericConfigElement driverConfig = new GenericConfigElement("driver");
            final GenericConfigElement localPathConfig = new GenericConfigElement("LocalPath");
            localPathConfig.setValue("/tmp");
            driverConfig.addChild(localPathConfig);
            DiskDeviceContext diskDeviceContext = (DiskDeviceContext) diskInterface.createContext(share, driverConfig);
            diskDeviceContext.setShareName(share);
            diskDeviceContext.setConfigurationParameters(driverConfig);
            diskDeviceContext.enableChangeHandler(false);
            diskDeviceContext.setDiskInformation(new SrvDiskInfo(2560000, 64, 512, 2304000));// Default to a 80Gb sized disk with 90% free space
            DiskSharedDevice diskDev = new DiskSharedDevice(share, diskInterface, diskDeviceContext);
            diskDev.setConfiguration(this);
            diskDev.setAccessControlList(secConfig.getGlobalAccessControls());
            diskDeviceContext.startFilesystem(diskDev);
            filesysConfig.addShare(diskDev);

            // SMB
            CIFSConfigSection cifsConfig = new CIFSConfigSection(this);

            if(broadcastMask != null && broadcastMask.length() > 0) {
                cifsConfig.setBroadcastMask(broadcastMask);
            }

            if(domain != null && domain.length() > 0) {
                cifsConfig.setDomainName(domain);
            }

            cifsConfig.setServerName(hostname);
            cifsConfig.setHostAnnounceInterval(5);
            cifsConfig.setHostAnnouncer(true);
            cifsConfig.setWin32NetBIOS(true);
            cifsConfig.setWin32HostAnnounceInterval(5);
            cifsConfig.setTcpipSMB(true);

            final EnterpriseCifsAuthenticator authenticator = new EnterpriseCifsAuthenticator();
            authenticator.setDebug(true);
            authenticator.setAllowGuest(false);
            authenticator.setAccessMode(EnterpriseCifsAuthenticator.NTLM2);
            authenticator.initialize(this, new GenericConfigElement("authenticator"));
            cifsConfig.setAuthenticator(authenticator);

            // For debugging SMB in depth
            //cifsConfig.setSessionDebugFlags(-1);
        }
    }
}
