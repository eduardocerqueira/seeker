//date: 2022-05-03T17:22:40Z
//url: https://api.github.com/gists/4c702c2ea59dabdd1c8fab5b1139ddbd
//owner: https://api.github.com/users/kirshiyin89

 @ArchTest
 static final ArchRule onion_architecture_is_respected = onionArchitecture()
            .domainModels("..domain.model..")
            .domainServices("..domain.service..")
            .applicationServices("..application..")
            .adapter("cli", "..adapter.cli..")
            .adapter("persistence", "..adapter.persistence..")
            .adapter("rest", "..adapter.rest..");