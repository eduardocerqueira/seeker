//date: 2024-05-15T16:59:05Z
//url: https://api.github.com/gists/1147e713c5c3164b8a8998b801a561d4
//owner: https://api.github.com/users/Vethursan

package com.thentiacloud.core.dropwizard.logging;
import com.thentiacloud.core.common.model.ThentiaCloudUser;
import com.thentiacloud.core.common.utility.Check;
import org.slf4j.Logger;

public class ThentiaCloudLogging
{
    //
    // Static In process Logging
    //
    public static void logError(Logger logger, String correlationId, ThentiaCloudUser user, Throwable exception)
    {
        if (Check.isSafe(logger) && (correlationId != null) && Check.isSafe(exception))
        {
            logger.error(String.format("[%1$s]\tUser: %2$s \tThentia Cloud Error occurred: %3$s", correlationId, getUserFullName(user), exception.getMessage()), exception);
        }
    }

    public static void logError(Logger logger, String correlationId, ThentiaCloudUser user, String errorMessage)
    {
        if (Check.isSafe(logger) && (correlationId != null) && Check.isSafe(errorMessage))
        {
            logger.error(String.format("[%1$s]\tUser: %2$s \tThentia Cloud Error occurred: %3$s", correlationId, getUserFullName(user), errorMessage));
        }
    }

    public static void logInfo(Logger logger, String correlationId, ThentiaCloudUser user, String message)
    {
        if (Check.isSafe(logger) && (correlationId != null) && Check.isSafe(message))
        {
            logger.info(String.format("[%1$s]\tUser: %2$s \tThentia Cloud information: %3$s", correlationId, getUserFullName(user), message));
        }
    }

    public static void logWarning(Logger logger, String correlationId, ThentiaCloudUser user, String message)
    {
        if (Check.isSafe(logger) && (correlationId != null) && Check.isSafe(message))
        {
            logger.warn(String.format("[%1$s]\tUser: %2$s \tThentia Cloud warning: %3$s", correlationId, getUserFullName(user), message));
        }
    }

    private static String getUserFullName(ThentiaCloudUser user)
    {
        return Check.isSafe(user) ? user.getFirstName() + " " + user.getLastName() : "Thentia Cloud System";
    }
}
