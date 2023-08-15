//date: 2023-08-15T16:48:38Z
//url: https://api.github.com/gists/c25fc093570fa6b8a7a67f21777600fb
//owner: https://api.github.com/users/shasisingh

 public static void main(String[] args) {
        var task = new TimerTask() {

            @Override
            public void run() {
                System.out.println("run some task");
            }
        };
        var timeer = new Timer();
        timeer.scheduleAtFixedRate(task, 10, 10);
    }