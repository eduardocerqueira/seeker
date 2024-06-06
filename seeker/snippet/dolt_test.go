//date: 2024-06-06T16:59:15Z
//url: https://api.github.com/gists/82c6e828c53dfc253218c0256c0eb1d1
//owner: https://api.github.com/users/arvidfm

// go get github.com/dolthub/dolt/go@main github.com/dolthub/dolt/go/gen/proto/dolt/services/eventsapi@main

func TestWithDB(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// start dolt and connect to existing database
	defer RunDolt(t, ctx, "my_database")()

	conn := getConnection()
	// read from and write to database and run tests here
}

func RunDolt(t *testing.T, ctx context.Context, dbName string) (stop func()) {
	ctx, cancel := context.WithCancel(ctx)
	controller := svcs.NewController()
	cfg := getDoltConfig("/path/to/dolt_db", "root", "toor", 3306)

	// run dolt server in goroutine
	var g errgroup.Group
	g.Go(func() error {
		dEnv := env.Load(ctx, env.GetCurrentUserHomeDir, filesys.LocalFS, doltdb.LocalDirDoltDB, doltversion.Version)
		startErr, stopErr := sqlserver.Serve(ctx, doltversion.Version, cfg, controller, dEnv)
		if startErr != nil {
			return startErr
		}
		return stopErr
	})

	// wait for server to start
	_ = controller.WaitForStart()

	// initialise connection
	connectToDatabase(cfg.Host(), cfg.Port(), cfg.User(), cfg.Password(), dbName)

	return func() {
		// reset database after test
		_, resetErr := getConnection().ExecContext(ctx, "CALL DOLT_RESET('--hard', 'HEAD')")
		cancel()
		controller.Stop()
		serverErr := g.Wait()

		if resetErr != nil {
			t.Fatal(resetErr)
		} else if serverErr != nil {
			t.Fatal(serverErr)
		}
	}
}

func getDoltConfig(dataDir, username, password string, port int) servercfg.ServerConfig {
	cfg := servercfg.DefaultServerConfig().(*servercfg.YAMLConfig)
	cfg.ListenerConfig.PortNumber = &port
	cfg.DataDirStr = &dataDir
	cfg.UserConfig.Name = &username
	cfg.UserConfig.Password = "**********"
	cfg.SystemVars_ = map[string]any{
		"innodb_autoinc_lock_mode": 1,
	}
	return cfg
}