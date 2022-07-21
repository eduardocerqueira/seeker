//date: 2022-07-21T16:50:51Z
//url: https://api.github.com/gists/6b8000843ad2f1bb7aba26536856423c
//owner: https://api.github.com/users/Sindhunayak23

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