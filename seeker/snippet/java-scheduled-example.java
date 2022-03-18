//date: 2022-03-18T16:56:36Z
//url: https://api.github.com/gists/b059f8ee95ced0b4f0f7ca620d1584c0
//owner: https://api.github.com/users/AndriiMaliuta

ScheduledExecutorService service = Executors.newScheduledThreadPool(3);
        service.scheduleAtFixedRate(() -> {
            System.out.println("hello");
        }, 0L, 5L, TimeUnit.SECONDS);