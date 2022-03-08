//date: 2022-03-08T17:11:57Z
//url: https://api.github.com/gists/96ee595bafcbd7d18476ef3981696500
//owner: https://api.github.com/users/dawnenakey

import java.util.ArrayList;

public class MobilePhone {
	
	private ArrayList<Contact> contactList = new ArrayList<Contact>();

	// Getters and setters
	public ArrayList<Contact> getContacts() {
		return contactList;
	}

	public void setContacts(ArrayList<Contact> contacts) {
		this.contactList = contacts;
	}
	
	// Methods
	public void printListOfContacts() {
		if(contactList.size() == 0) {
			System.out.println("List of contacts is empty");
		} else {
			System.out.println("/*--- List of contacts ---*/");
			for(int i = 0; i < contactList.size(); i++) {
				System.out.println(contactList.get(i).getName() + ": " + contactList.get(i).getNumber());
			}
		}
	}
	
	public void addContact(Contact contact) {
		contactList.add(contact);
		System.out.println("Contact added");
	}
	
	public void updateContact(String name, Contact newContact) {
		int position = findContact(name);
		if(position < 0) {
			System.out.println("Contact not found");
		}
		contactList.set(position, newContact);
		System.out.println("Contact updated");
	}

	public void removeContact(String name) {
		int position = findContact(name);
		if(position < 0) {
			System.out.println("Contact not found");
		}
		contactList.remove(position);
		System.out.println("Contact deleted");
	}
	
	public int findContact(Contact contact) {
		return contactList.indexOf(contact);
	}
	
	public int findContact(String contactName) {
		for(int i = 0; i < contactList.size(); i++) {
			Contact contact = contactList.get(i);
			if(contact.getName().toLowerCase().equals(contactName.toLowerCase())) {
				return i;
			}
		}
		return -1;
	}
	
}