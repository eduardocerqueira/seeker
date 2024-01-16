//date: 2024-01-16T16:48:50Z
//url: https://api.github.com/gists/5b5f7dea48dfb660f4d97d8bd99be006
//owner: https://api.github.com/users/surli

/*
 * See the NOTICE file distributed with this work for additional
 * information regarding copyright ownership.
 *
 * This is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of
 * the License, or (at your option) any later version.
 *
 * This software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this software; if not, write to the Free
 * Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
 * 02110-1301 USA, or see the FSF site: http://www.fsf.org.
 */
package org.xwiki.notifications.filters.migration;

import java.sql.Connection;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.List;

import javax.inject.Inject;
import javax.inject.Provider;
import javax.inject.Singleton;

import org.hibernate.Session;
import org.hibernate.jdbc.Work;
import org.slf4j.Logger;
import org.xwiki.component.annotation.Component;
import org.xwiki.notifications.filters.internal.DefaultNotificationFilterPreference;
import org.xwiki.wiki.descriptor.WikiDescriptorManager;

import com.xpn.xwiki.XWikiContext;
import com.xpn.xwiki.XWikiException;
import com.xpn.xwiki.internal.store.hibernate.HibernateStore;
import com.xpn.xwiki.store.DatabaseProduct;
import com.xpn.xwiki.store.XWikiHibernateBaseStore;
import com.xpn.xwiki.store.XWikiHibernateStore;

@Component(roles = NotificationFiltersRecovery.class)
@Singleton
public class NotificationFiltersRecovery
{
    @Inject
    private Provider<XWikiContext> contextProvider;

    @Inject
    private WikiDescriptorManager wikiDescriptorManager;

    @Inject
    private HibernateStore hibernateStore;

    @Inject
    private Logger logger;

    public void recoverFrom(String dbName) throws Exception
    {
        XWikiContext context = contextProvider.get();
        XWikiHibernateStore hibernateStore = context.getWiki().getHibernateStore();
        String statement = "select nfp "
            + "from DefaultNotificationFilterPreference nfp "
            + "where length(nfp.pageOnly) > 0 "
            + "order by nfp.id ASC ";

        int offset = 0;
        List<DefaultNotificationFilterPreference> notificationFilterPreferences;
        String mainWikiId = this.wikiDescriptorManager.getMainWikiId();

        do {
            int finalOffset = offset;
            notificationFilterPreferences = executeRead(hibernateStore, dbName,
                    session -> session.createQuery(statement).setMaxResults(1000).setFirstResult(finalOffset).list());

            this.logger.info("Found [{}] filters to analyze for recovery...", notificationFilterPreferences.size());

            int i = 0;
            for (DefaultNotificationFilterPreference preference : notificationFilterPreferences) {
                if (!filterStillExists(hibernateStore, mainWikiId, preference)) {
                    hibernateStore.executeWrite(context, session -> {
                        DefaultNotificationFilterPreference copy =
                            new DefaultNotificationFilterPreference(preference, false);
                        copy.setOwner(preference.getOwner());
                        session.saveOrUpdate(copy);
                        return null;
                    });
                    i++;
                }
            }
            this.logger.info("[{}] filters have been recovered.", i);
            offset += 1000;
        } while (!notificationFilterPreferences.isEmpty());

    }

    private boolean filterStillExists(XWikiHibernateStore hibernateStore, String mainwiki,
        DefaultNotificationFilterPreference filterPreference) throws XWikiException
    {
        XWikiContext context = contextProvider.get();
        String existenceStatement = "select count(nfp) "
            + "from DefaultNotificationFilterPreference nfp "
            + "where nfp.pageOnly = :pageOnly and nfp.owner = :owner and nfp.filterName = :filterName "
            + "and nfp.startingDate = :startingDate ";
        List<Long> result = executeRead(hibernateStore, mainwiki, session ->
            session.createQuery(existenceStatement)
                .setParameter("pageOnly", filterPreference.getPageOnly())
                .setParameter("owner", filterPreference.getOwner())
                .setParameter("filterName", filterPreference.getFilterName())
                .setParameter("startingDate", filterPreference.getStartingDate())
                .setMaxResults(1)
                .list());
        return result.get(0) > 0;
    }

    private <T> T executeRead(XWikiHibernateStore xWikiHibernateStore,
        String database, XWikiHibernateBaseStore.HibernateCallback<T> cb)
        throws XWikiException
    {
        Session session = xWikiHibernateStore.getSessionFactory().openSession();
        this.setDatabase(database, session);
        try {
            // Execute the callback
            T result = cb.doInHibernate(session);
            return result;
        } catch (XWikiException e) {
            throw e;
        } catch (Exception e) {
            throw new XWikiException(XWikiException.MODULE_XWIKI_STORE, XWikiException.ERROR_XWIKI_UNKNOWN,
                "Exception while hibernate execute", e);
        } finally {
            session.close();
        }
    }

    private void setDatabase(String dbName, Session session)
    {
        DatabaseProduct product = this.hibernateStore.getDatabaseProductName();
        if (DatabaseProduct.ORACLE == product) {
            executeStatement("alter session set current_schema = " + dbName, session);
        } else if (DatabaseProduct.DERBY == product || DatabaseProduct.HSQLDB == product
            || DatabaseProduct.DB2 == product || DatabaseProduct.H2 == product) {
            executeStatement("SET SCHEMA " + dbName, session);
        } else if (DatabaseProduct.POSTGRESQL == product && this.hibernateStore.isConfiguredInSchemaMode()) {
            executeStatement("SET search_path TO " + dbName, session);
        } else {
            session.doWork(connection -> {
                String catalog = connection.getCatalog();
                catalog = (catalog == null) ? null : catalog.replace('_', '-');
                if (!dbName.equals(catalog)) {
                    connection.setCatalog(dbName);
                }
            });
        }

        session.setProperty("xwiki.database", dbName);
    }

    private void executeStatement(final String sql, Session session)
    {
        session.doWork(new Work()
        {
            @Override
            public void execute(Connection connection) throws SQLException
            {
                try (Statement stmt = connection.createStatement()) {
                    stmt.execute(sql);
                }
            }
        });
    }
}
