//date: 2024-01-03T16:48:34Z
//url: https://api.github.com/gists/6c058dcbb26cad9f54090179d33af1e5
//owner: https://api.github.com/users/SA-JackMax

/*
 Copyright 2016 Jared Ottley <jared@ottleys.net>
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
  http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

package org.alfresco.extension.migration;

import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.alfresco.cmis.client.AlfrescoDocument;
import org.alfresco.cmis.client.AlfrescoFolder;
import org.alfresco.extension.migration.util.CodeUrl;

import org.apache.chemistry.opencmis.client.api.CmisObject;
import org.apache.chemistry.opencmis.client.api.FileableCmisObject;
import org.apache.chemistry.opencmis.client.api.Folder;
import org.apache.chemistry.opencmis.client.api.ItemIterable;
import org.apache.chemistry.opencmis.client.api.OperationContext;
import org.apache.chemistry.opencmis.client.api.QueryResult;
import org.apache.chemistry.opencmis.client.api.Repository;
import org.apache.chemistry.opencmis.client.api.Session;
import org.apache.chemistry.opencmis.client.api.SessionFactory;
import org.apache.chemistry.opencmis.client.api.Tree;
import org.apache.chemistry.opencmis.client.runtime.OperationContextImpl;
import org.apache.chemistry.opencmis.client.runtime.SessionFactoryImpl;
import org.apache.chemistry.opencmis.commons.PropertyIds;
import org.apache.chemistry.opencmis.commons.SessionParameter;
import org.apache.chemistry.opencmis.commons.enums.BaseTypeId;
import org.apache.chemistry.opencmis.commons.enums.BindingType;
import org.apache.chemistry.opencmis.commons.enums.VersioningState;
import org.apache.chemistry.opencmis.commons.exceptions.CmisContentAlreadyExistsException;
import org.apache.chemistry.opencmis.commons.exceptions.CmisRuntimeException;

import org.mortbay.jetty.Server;
import org.mortbay.jetty.handler.RewriteHandler;

import org.openqa.selenium.WebElement;
import org.openqa.selenium.htmlunit.HtmlUnitDriver;

import org.springframework.social.alfresco.api.Alfresco;
import org.springframework.social.alfresco.connect.AlfrescoConnectionFactory;
import org.springframework.social.connect.Connection;
import org.springframework.social.oauth2.AccessGrant;
import org.springframework.social.oauth2.GrantType;
import org.springframework.social.oauth2.OAuth2Parameters;

/*
* @author Jared Ottley <jottley@gmail.com>
*/

public class Community2

{
    private SessionFactory sessionFactory;
    private Session session;
    private static Session _session;

    private String username;
    private String password;

    private String alfrescoHost;

    public static final int PAGINATE_MAX_ITEMS = 10;

    private OperationContext paginateContext;

    public static final String SITES_QUERY = “SELECT * FROM st:site WHERE cmis:name = 'deacon'”;

    private static final String CONSUMER_KEY = “<CONSUMER_KEY_HERE>”;
    private static final String CONSUMER_SECRET = "**********"

    private static final String REDIRECT_URI = “http://localhost:9876”;

    private static final String STATE = “site-migration”;

    private static Server server;

    private static AlfrescoConnectionFactory connectionFactory;
    private static Connection<Alfresco> connection;

    private static org.alfresco.extension.migration.util.AuthUrl authUrlObject;

    private static AccessGrant accessGrant;
    private static Alfresco alfresco;
    private static Community2 community;

    protected static long documents = 0L;
    protected static long folders = 0L;

    public static void main(String[] args)
    {
        try
        {
            System.out.println(“Initializing Migration Utility…”);
            community = new Community2();
            community.setAlfrescoHost(“http://localhost:8080”);
            community.setUsername(“admin”);
            community.setPassword(“admin”);

            System.out.printf(“ Connecting to Alfresco Community…”);
            community.connectToRepository();
            AlfrescoFolder communitySite = community.getSite().get(0);

            System.out.printf(“Connected!%n”);
            // System.out.printf(“ Connecting to Alfresco Cloud…”);
            // community.connectToCloud();
            // List<AlfrescoFolder> cloudSite = community.getCloudSite();
            // AlfrescoFolder documentLibrary = community.getDocLib(cloudSite.get(0));
            
            // System.out.printf(“Connected!%n”);
            // System.out.printf(“Initialization Complete!%n%n”);
            // System.out.println(“=============================================”);
            // System.out.println(“Starting Migration….”);

            // community.doWork(communitySite, documentLibrary);
            // System.out.println();
            // System.out.println(“=============================================”);
            // System.out.println(“Migration Complete! ” + Community2.documents + “ documents and ” + Community2.folders + “ folders migrated to Cloud.”);

            List<Tree<FileableCmisObject>> children = communitySite.getFolderTree(10);

            for (Tree<FileableCmisObject> tree : children)
            {
                if (tree.getItem().getName().equals(“documentLibrary”))
                {
                    System.out.println(tree.getItem().getName());

                    ItemIterable<CmisObject> documents = ((AlfrescoFolder)tree.getItem()).getChildren();

                    for (CmisObject document : documents)
                    {
                        System.out.println(document.getName());
                    }

                    List<Tree<FileableCmisObject>> subFolder = tree.getChildren();

                    for (Tree<FileableCmisObject> cmisObject : subFolder)
                    {
                        System.out.println(cmisObject.getItem().getName());
                    }
                }
            }
        }
        finally
        {
            try
            {
                // server.stop();
            }

            catch (Exception e)
            {
                // TODO Auto-generated catch block
                e.printStackTrace();
            }
        }
    }

    public void setUsername(String username)
    {
        this.username = username;
    }

    public void setPassword(String password)
    {
        this.password = "**********"
    }

    public void setAlfrescoHost(String alfrescoHost)
    {
        this.alfrescoHost = alfrescoHost;
    }

    public void connectToRepository()
    {
        // default factory implementation
        sessionFactory = SessionFactoryImpl.newInstance();
        Map<String, String> parameter = new HashMap<String, String>();

        // connection settings
        parameter.put(SessionParameter.ATOMPUB_URL, alfrescoHost + “/alfresco/cmisatom”);
        parameter.put(SessionParameter.BINDING_TYPE, BindingType.ATOMPUB.value());

        // user credentials
        parameter.put(SessionParameter.USER, username);
        parameter.put(SessionParameter.PASSWORD, password);

        // Set the alfresco object factory
        parameter.put(SessionParameter.OBJECT_FACTORY_CLASS, “org.alfresco.cmis.client.impl.AlfrescoObjectFactoryImpl”);

        // create session
        Repository repository = sessionFactory.getRepositories(parameter).get(0);
        session = repository.createSession();

        // Default Pagination
        paginateContext = new OperationContextImpl();
        paginateContext.setMaxItemsPerPage(Community2.PAGINATE_MAX_ITEMS);
        paginateContext.setOrderBy(“cmis:baseTypeId ASC”);
    }

    public Session getSession()
    {
        if (session == null)
        {
            throw new CmisRuntimeException(“Session not initialized”)
        }

        return session;
    }

    public List<AlfrescoFolder> getSite()
    {
        List<AlfrescoFolder> sites = new ArrayList<AlfrescoFolder>();
        int count = 0;
        boolean more = true;

        while (more)
        {
            try
            {
                ItemIterable<QueryResult> paginatedResults = session.query(SITES_QUERY, false, paginateContext).skipTo(count).getPage();
                Iterator<QueryResult> pageItems = paginatedResults.iterator();

                while (pageItems.hasNext())
                {
                    QueryResult queryResult = pageItems.next();
                    sites.add((AlfrescoFolder)session.getObject(queryResult.getPropertyValueById(“cmis:objectId”).toString()));
                }

                count = count + Community2.PAGINATE_MAX_ITEMS;
                more = paginatedResults.getHasMoreItems();
            }
            catch (CmisRuntimeException e)
            {
                // Place Holder
            }
        }

        return sites;
    }

    public List<AlfrescoFolder> getCloudSite()
    {
        List<AlfrescoFolder> sites = new ArrayList<AlfrescoFolder>();
        int count = 0;
        boolean more = true;

        while (more)
        {
            try
            {
                ItemIterable<QueryResult> paginatedResults = _session.query(SITES_QUERY, false, paginateContext).skipTo(count).getPage();
                Iterator<QueryResult> pageItems = paginatedResults.iterator();

                while (pageItems.hasNext())
                {
                    QueryResult queryResult = pageItems.next();
                    sites.add((AlfrescoFolder)_session.getObject(queryResult.getPropertyValueById(“cmis:objectId”).toString()));
                }

                count = count + Community2.PAGINATE_MAX_ITEMS;
                more = paginatedResults.getHasMoreItems();
            }
            catch (CmisRuntimeException e)
            {
                // Place Holder
            }
        }

        return sites;
    }

    public ItemIterable<CmisObject> getChildren(AlfrescoFolder folder)
    {
        ItemIterable<CmisObject> children = folder.getChildren();

        if (children.getTotalNumItems() == 1)
        {
            for (CmisObject cmisObject : children)
            {
                if (cmisObject.getName().equals(“documentLibrary”))
                {
                    children = ((AlfrescoFolder)cmisObject).getChildren();
                }
            }
        }

        return children;
    }

    private AlfrescoFolder getDocLib(AlfrescoFolder site)
    {
        AlfrescoFolder doclib = null;
        ItemIterable<CmisObject> children = site.getChildren();

        if (children.getTotalNumItems() == 1)
        {
            for (CmisObject cmisObject : children)
            {
                if (cmisObject.getName().equals(“documentLibrary”))
                {
                    doclib = (AlfrescoFolder)cmisObject;
                }
            }
        }

        return doclib;
    }

    private void migrateDocument(AlfrescoFolder parent, AlfrescoDocument document)
    {
        System.out.println(“Document Found: ” + document.getName());

        try
        {
            Map<String, Object> properties = new HashMap<String, Object>();
            properties.put(PropertyIds.OBJECT_TYPE_ID, “cmis:document”);
            properties.put(PropertyIds.NAME, document.getName());

            System.out.printf(“Migrating to cloud…”);;
            parent.createDocument(properties, document.getContentStream(), VersioningState.MAJOR);

            Community2.documents++;

            System.out.printf(“Migration successful!%n”);
        }
        catch (CmisContentAlreadyExistsException e)
        {
            System.out.println(e.getMessage());
        }
    }

    private AlfrescoFolder createFolder(AlfrescoFolder parent, AlfrescoFolder folder)
    {
        System.out.println(“Folder Found: ” + folder.getName());
        Folder newFolder = null;

        try
        {
            Map<String, Object> properties = new HashMap<String, Object>();
            properties.put(PropertyIds.OBJECT_TYPE_ID, “cmis:folder”);
            properties.put(PropertyIds.NAME, folder.getName());

            System.out.printf(“Creating folder in cloud…”);
            newFolder = parent.createFolder(properties);
            Community2.folders++;

            System.out.printf(“Creation successful!%n”);
        }
        catch (CmisContentAlreadyExistsException e)
        {
            System.out.println(e.getMessage());
            newFolder = community.getFolder(parent, folder.getName());
        }

        return (AlfrescoFolder)newFolder;
    }

    private void doWork(AlfrescoFolder sourceParent, AlfrescoFolder targetParent)
    {
        ItemIterable<CmisObject> children = community.getChildren(sourceParent);

        for (CmisObject cmisObject : children)
        {
            if (cmisObject.getBaseTypeId().equals(BaseTypeId.CMIS_DOCUMENT))
            {
                community.migrateDocument(targetParent, (AlfrescoDocument)cmisObject);
            }
            else if (cmisObject.getBaseTypeId().equals(BaseTypeId.CMIS_FOLDER))
            {
                AlfrescoFolder newFolder = community.createFolder(targetParent, (AlfrescoFolder)cmisObject);

                doWork((AlfrescoFolder)cmisObject, newFolder);
            }
        }
    }

    private AlfrescoFolder getFolder(AlfrescoFolder parent, String name)
    {
        ItemIterable<CmisObject> children = parent.getChildren();

        for (CmisObject cmisObject : children)
        {
            if (cmisObject.getName().equals(name))
            {
                return (AlfrescoFolder)cmisObject;
            }
        }

        return null;
    }

    public void connectToCloud()
    {
        try
        {
            setupServer();
            authenticate();

            GetAPI(“<username>”, “<password>”);
            _session = alfresco.getCMISSession(“<alfresco tennant>”);
        }
        catch (IOException e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        catch (Exception e)
        {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    private void setupServer()
        throws Exception
    {
        server = new Server(9876);
        server.setHandler(new RewriteHandler());
        server.start();
    }

    private void authenticate()
        throws MalformedURLException
    {
        connectionFactory = "**********"

        OAuth2Parameters parameters = new OAuth2Parameters();
        parameters.setRedirectUri(REDIRECT_URI);
        parameters.setScope(Alfresco.DEFAULT_SCOPE);
        parameters.setState(STATE);

        authUrlObject = new org.alfresco.extension.migration.util.AuthUrl(connectionFactory.getOAuthOperations().buildAuthenticateUrl(GrantType.AUTHORIZATION_CODE, parameters));
    }

    private void GetAPI(String username, String password)
        throws IOException
    {
        HtmlUnitDriver driver = new HtmlUnitDriver();
        driver.get(authUrlObject.toString());

        List<WebElement> webElements = driver.findElementsByTagName(“form”);
        WebElement usernameElement = driver.findElementById(“username”);
        usernameElement.sendKeys(username);

        WebElement passwordElement = "**********"
        passwordElement.sendKeys(password);
        webElements.get(0).submit();

        CodeUrl codeUrl = new CodeUrl(driver.getCurrentUrl());
        accessGrant = connectionFactory.getOAuthOperations().exchangeForAccess(codeUrl.getQueryMap().get(CodeUrl.CODE), REDIRECT_URI, null);
        connection = connectionFactory.createConnection(accessGrant);

        alfresco = connection.getApi();
    }
}ionFactory.createConnection(accessGrant);

        alfresco = connection.getApi();
    }
}