//date: 2022-11-11T17:02:09Z
//url: https://api.github.com/gists/d8bfa2128844cefd032b2d6892f55e17
//owner: https://api.github.com/users/gushakov

    @Builder
    public Delivery(TransportStatus transportStatus, UnLocode lastKnownLocation, VoyageNumber currentVoyage,
                    UtcDateTime eta, RoutingStatus routingStatus, boolean misdirected,
                    HandlingActivity nextExpectedActivity) {
        this.transportStatus = notNull(transportStatus);
        this.lastKnownLocation = lastKnownLocation;
        this.currentVoyage = currentVoyage;
        this.lastEvent = null;
        this.eta = eta;
        this.routingStatus = notNull(routingStatus);
        this.misdirected = misdirected;
        this.nextExpectedActivity = nextExpectedActivity;
    }
