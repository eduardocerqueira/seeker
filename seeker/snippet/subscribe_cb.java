//date: 2023-02-13T17:04:30Z
//url: https://api.github.com/gists/4f0e1c4585b1a844356a7160166e3557
//owner: https://api.github.com/users/Bravo27

    public void subscribe(String itemName, boolean needsIterator) throws SubscriptionException, FailureException {

        logger.info("Subscribe for item: " + itemName);

        if (itemName.startsWith("DepartureMonitor")) {
            consumer = new ConsumerLoop(this, kconnstring, kconsumergroupid, ktopicname);
            consumer.start();
        } else if (itemName.startsWith("CurrTime")) {
            currtime = true;
        } else {
            logger.warn("Requested item not expected.");
        }
        
    }