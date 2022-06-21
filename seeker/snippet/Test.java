//date: 2022-06-21T17:01:04Z
//url: https://api.github.com/gists/20471d629ee17df904ede98ceb849172
//owner: https://api.github.com/users/mmoayyed

@Test
    public void verifyCompatibilityWithRegex() {
        val service = new RegexRegisteredService();
        service.setId(2020);
        service.setServiceId("http://localhost:8080");
        service.setName("Testing");
        service.setDescription("Testing Application");
        service.setTheme("theme");
        service.setAttributeReleasePolicy(new ReturnAllAttributeReleasePolicy());
        val accessStrategy = new DefaultRegisteredServiceAccessStrategy();
        accessStrategy.setDelegatedAuthenticationPolicy(new DefaultRegisteredServiceDelegatedAuthenticationPolicy()
            .setAllowedProviders(CollectionUtils.wrapList("one", "two"))
            .setPermitUndefined(false)
            .setExclusive(false));
        service.setMultifactorAuthenticationPolicy(new DefaultRegisteredServiceMultifactorPolicy()
            .setMultifactorAuthenticationProviders(CollectionUtils.wrapSet("one", "two")));
        service.setAccessStrategy(accessStrategy);
        newServiceRegistry.save(service);
        val services = newServiceRegistry.load();
        assertEquals(1, services.size());
    }