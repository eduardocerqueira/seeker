//date: 2023-01-06T16:58:38Z
//url: https://api.github.com/gists/62b3869518c5a060d006166b305068da
//owner: https://api.github.com/users/stanio

/*
 * This module, both source code and documentation,
 * is in the Public Domain, and comes with NO WARRANTY.
 */
package io.github.stanio.imageio;

import static javax.xml.XMLConstants.XMLNS_ATTRIBUTE;

import java.io.IOException;
import java.io.StringReader;

import javax.xml.XMLConstants;
import javax.xml.parsers.FactoryConfigurationError;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.InputSource;
import org.xml.sax.Locator;
import org.xml.sax.SAXException;
import org.xml.sax.SAXNotRecognizedException;
import org.xml.sax.SAXNotSupportedException;
import org.xml.sax.XMLReader;
import org.xml.sax.ext.DefaultHandler2;
import org.xml.sax.ext.Locator2;
import org.xml.sax.helpers.AttributesImpl;

/**
 * Encapsulates the prolog ({@code xml} and {@code DOCTYPE} declarations, if
 * any) and root element information of an XML document used to identify its
 * content type.
 */
public class XMLDoctype {

    private static ThreadLocal<XMLPrologHandler>
            localHandler = new ThreadLocal<XMLPrologHandler>() {
        @Override protected XMLPrologHandler initialValue() {
            return new XMLPrologHandler();
        }
    };

    String xmlVersion;
    String encoding;
    String name;
    String publicId;
    String systemId;
    String rootQName;
    Attributes rootAttributes;

    public XMLDoctype() {
        // empty
    }

    public static XMLDoctype of(InputSource source) throws IOException, SAXException {
        // REVISIT: Unlikely scenario but limit the source data, so we don't end up reading megabytes
        // of misc. content (comments and PIs) before, if ever reaching root document element.
        return localHandler.get().parse(source);
    }

    public static XMLDoctype of(String url) throws IOException, SAXException {
        return of(new InputSource(url));
    }

    public String getXmlVersion() {
        return xmlVersion;
    }

    public String getEncoding() {
        return encoding;
    }

    public String getName() {
        return name;
    }

    public String getPublicId() {
        return publicId;
    }

    public String getSystemId() {
        return systemId;
    }

    public String getRootNamespace() {
        int colonIndex = rootQName.indexOf(':');
        if (colonIndex < 0) {
            return rootAttributes.getValue(XMLNS_ATTRIBUTE);
        }
        return rootAttributes.getValue(XMLNS_ATTRIBUTE
                + ':' + rootQName.substring(0, colonIndex));
    }

    public String getRootLocalName() {
        int colonIndex = rootQName.lastIndexOf(':');
        return colonIndex < 0 ? rootQName
                              : rootQName.substring(colonIndex + 1);
    }

    public String getRootQName() {
        return rootQName;
    }

    public Attributes getRootAttributes() {
        return rootAttributes;
    }

    @Override
    public String toString() {
        return "Doctype(xmlVersion=" + xmlVersion
                + ", encoding=" + encoding
                + ", name=" + name
                + ", publicId=" + publicId
                + ", systemId=" + systemId
                + ", rootQName=" + rootQName
                + ", rootAttributes=" + rootAttributes + ")";
    }

} // class XMLDoctype


class XMLPrologHandler extends DefaultHandler2 {

    private static SAXParserFactory saxParserFactory;

    private XMLDoctype doctype;
    private Locator locator;
    private XMLReader xmlReader;

    public XMLPrologHandler() {
        try {
            synchronized (XMLPrologHandler.class) {
                xmlReader = saxParserFactory().newSAXParser().getXMLReader();
            }
        } catch (SAXException | ParserConfigurationException e) {
            throw new IllegalStateException(e);
        }

        xmlReader.setContentHandler(this);
        xmlReader.setErrorHandler(this);
        xmlReader.setEntityResolver(this);
        try {
            xmlReader.setProperty("http://xml.org/sax/properties/"
                                  + "lexical-handler", this);
        } catch (SAXNotRecognizedException | SAXNotSupportedException e) {
            // Optional
        }
    }

    private static SAXParserFactory saxParserFactory() {
        if (saxParserFactory == null) {
            try {
                SAXParserFactory spf = SAXParserFactory.newInstance();
                spf.setNamespaceAware(false);
                spf.setValidating(false);
                spf.setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true);
                saxParserFactory = spf;
            } catch (SAXException | ParserConfigurationException e) {
                throw new FactoryConfigurationError(e);
            }
        }
        return saxParserFactory;
    }

    public XMLDoctype parse(InputSource source) throws IOException, SAXException {
        XMLDoctype doctype = this.doctype = new XMLDoctype();
        try {
            xmlReader.parse(source);
        } catch (StopParseException e) {
            // Found root element
        } finally {
            this.locator = null;
            this.doctype = null;
        }
        return doctype;
    }

    @Override
    public void setDocumentLocator(Locator locator) {
        this.locator = locator;
    }

    @Override
    public void startDTD(String name, String publicId, String systemId)
            throws SAXException {
        doctype.name = name;
        doctype.publicId = publicId;
        doctype.systemId = systemId;
    }

    @Override
    public void startElement(String uri,
                             String localName,
                             String qName,
                             Attributes attributes)
            throws SAXException {
        doctype.rootQName = qName;
        doctype.rootAttributes = new AttributesImpl(attributes);

        if (locator instanceof Locator2) {
            Locator2 locator2 = (Locator2) locator;
            doctype.xmlVersion = locator2.getXMLVersion();
            doctype.encoding = locator2.getEncoding();
        }

        throw StopParseException.INSTANCE;
    }

    @Override
    public InputSource resolveEntity(String name,
                                     String publicId,
                                     String baseURI,
                                     String systemId) {
        // Don't resolve any external entities – just replace with empty
        // content.  A more general accessExternalDTD="" setup.
        return new InputSource(new StringReader(""));
    }


    @SuppressWarnings("serial")
    private static class StopParseException extends SAXException {

        static final StopParseException INSTANCE = new StopParseException();

        private StopParseException() {
            super("Parsing stopped from content handler");
        }

        @Override
        public synchronized Throwable fillInStackTrace() {
            return this; // Don't fill in stack trace
        }

    } // class StopParseException


} // class XMLPrologHandler
