//date: 2022-12-22T16:44:06Z
//url: https://api.github.com/gists/5c888428c2c7bce490f2b42adc7cc1b5
//owner: https://api.github.com/users/AnkitGuptahbti

                //login 
		//setting up parameters for login method
		User_auth auth = new User_auth();
		auth.setUser_name(USER_NAME);
		auth.setPassword(PASSWORD);
		
		//sending an empty name_value_list
		Name_value nameValueListLogin[] = null;
		
		//trying to login
		Entry_value loginResponse = null;
		try {
			loginResponse = stub.login(auth, APPLICATION_NAME , nameValueListLogin);
		} catch (RemoteException e) {
			System.out.println("login failed. Message: "+e.getMessage());
			e.printStackTrace();
		}
		System.out.println("login successful! login id: "+loginResponse.getId());