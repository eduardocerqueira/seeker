//date: 2023-02-08T17:02:06Z
//url: https://api.github.com/gists/7aa7b14c241c9bafd725ab383c98ed48
//owner: https://api.github.com/users/COdErJ26

String msg = "Android : ";
    /** Called when the activity is first created. */
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main); //loads the ui component from activity_main.xml

    }
        @Override
        protected void onStart() {
            super.onStart();

        }

        /** Called when the activity has become visible. */
        @Override
        protected void onResume() {
            super.onResume();
            Log.d(msg, "The onResume() event");
        }

        /** Called when another activity is taking focus. */
        @Override
        protected void onPause() {
            super.onPause();
            Log.d(msg, "The onPause() event");
        }

        /** Called when the activity is no longer visible. */
        @Override
        protected void onStop() {
            super.onStop();
            Log.d(msg, "The onStop() event");
        }

        /** Called just before the activity is destroyed. */
        @Override
        public void onDestroy() {
            super.onDestroy();
            Log.d(msg, "The onDestroy() event");
        }