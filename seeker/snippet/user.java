//date: 2022-01-25T16:54:23Z
//url: https://api.github.com/gists/4916effbee5b98266378ae1cf1de0287
//owner: https://api.github.com/users/YonathanMeguira

/** here is the model sync service is expecting to receive.**/
package com.datorama.core.dto.admin.gdo;
import java.util.Collection;
import com.datorama.core.dto.admin.UserDto;
public class GdoUser {
	private final Integer id;
	private final String email;
	private final String firstName;
	private final String lastName;
	private final Integer language;
	private final Collection<GdoUserAccount> accounts;
	private final String userDataJson;
	public GdoUser(UserDto user, Collection<GdoUserAccount> accounts) {
		this.id = user.getId();
		this.firstName = user.getFirstName();
		this.email = user.getEmail();
		this.lastName = user.getLastName();
		this.language = user.getLanguageId();
		this.accounts = accounts;
		this.userDataJson = user.getJsonConfig();
	}
	public Integer getId() {
		return id;
	}
	public String getEmail() {
		return email;
	}
	
	public String getFirstName() {
		return firstName;
	}
	
	public String getLastName() {
		return lastName;
	}
	
	public Integer getLanguage() {
		return language;
	}
	
	public Collection<GdoUserAccount> getAccounts() {
		return accounts;
	}
	
	public String getUserJsonConfig() {
		return userDataJson;
	}
	
}
package com.datorama.core.dto.admin.gdo;
public class GdoUserAccount {
	private final Integer accountId;
	private final boolean isOriginalAccount;
	public GdoUserAccount(Integer accountId, boolean isOriginalAccount) {
		this.accountId = accountId;
		this.isOriginalAccount = isOriginalAccount;
	}
	
	public Integer getAccountId() {
		return accountId;
	}
	public boolean isOriginalAccount() {
		return isOriginalAccount;
	}
}