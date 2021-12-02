//date: 2021-12-02T17:08:40Z
//url: https://api.github.com/gists/d3dcd94df44fbae2c5f9e5505f7b2731
//owner: https://api.github.com/users/jitendraavinash

package com.pattern.creational;

public class FactoryMethod {
    public static void main(String[] args) {
        NotificationService ns;
        Notifier notifier = null;
        String notificationType = args[0];
        if (notificationType.equals("email")) {
            notifier = new EmailService(notificationType);
        }
        if (notificationType.equals("sms")) {
            notifier = new SmsService(notificationType);
        }
        assert notifier != null;
        ns = notifier.notifyUser();
        System.out.println(ns);
    }
}

interface Notifier {
    NotificationService notifyUser();
}

class NotificationService {
    private String notificationType;

    public NotificationService(String notificationType) {
        this.notificationType = notificationType;
    }
}

class EmailService implements Notifier {
    private NotificationService ns;

    public EmailService(String notificationType) {
        this.ns = new NotificationService(notificationType);
    }

    @Override
    public NotificationService notifyUser() {
        return ns;
    }
}

class SmsService implements Notifier {
    private NotificationService ns;

    public SmsService(String notificationType) {
        ns = new NotificationService(notificationType);
    }

    @Override
    public NotificationService notifyUser() {
        return ns;
    }
}
