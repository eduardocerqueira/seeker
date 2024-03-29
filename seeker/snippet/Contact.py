#date: 2024-03-29T16:59:19Z
#url: https://api.github.com/gists/65d0a42f107c0f99b7d17905437eaf84
#owner: https://api.github.com/users/hernamesbarbara

#!/usr/bin/env python3
from datetime import datetime
import objc
from Contacts import (CNMutableContact, CNContactStore, CNSaveRequest, CNLabeledValue,
                      CNPhoneNumber, CNLabelURLAddressHomePage)
from Foundation import NSCalendar, NSDateComponents


class Contact:
    """
    Simple wrapper for Apple iOS / macOS Contact record
    """

    def __init__(self, first_name, last_name, email, phone, job_title, company, website, note=""):
        """
        Initialize a Contact object with the given parameters.

        Parameters:
            first_name (str)
            last_name (str)
            email (str): Valid email
            phone (str): will accept punctuation and/or spaces
            job_title (str)
            company (str)
            website (str)
            note (str, optional): Additional note for the contact. Defaults to "".

        NOTE:
            The `note` property doesnt work right now. There is a bug. See inline comments about it.
        """
        self.contact = CNMutableContact.new()
        self.contact.setGivenName_(first_name)
        self.contact.setFamilyName_(last_name)
        self.contact.setJobTitle_(job_title)
        self.contact.setOrganizationName_(company)

        # Set email and phone with their types
        # Look at Apple's Objective-C docs for valid / acceptable label types
        emailValue = CNLabeledValue.alloc().initWithLabel_value_("work", email)
        self.contact.setEmailAddresses_([emailValue])
        phoneValue = CNLabeledValue.alloc().initWithLabel_value_(
            "mobile", CNPhoneNumber.phoneNumberWithStringValue_(phone))
        self.contact.setPhoneNumbers_([phoneValue])

        if website:
            websiteValue = CNLabeledValue.alloc().initWithLabel_value_(
                CNLabelURLAddressHomePage, website)
            self.contact.setUrlAddresses_([websiteValue])

        # TODO uncomment the next line and the script will fail
        # self.contact.setNote_("jodie")     # ¡¡ BROKEN NEED TO FIX !!

        # there is no visible created date in Apple Contacts.app by default
        # this gives you a way to see it...a bit of a hack
        today = datetime.now()

        dateComponents = NSDateComponents.alloc().init()
        dateComponents.setYear_(today.year)
        dateComponents.setMonth_(today.month)
        dateComponents.setDay_(today.day)

        customDateValue = CNLabeledValue.alloc().initWithLabel_value_(
            "created_date", dateComponents)
        self.contact.setDates_([customDateValue])

    def save(self):
        """
        Try to save the contact to Apple address book. 
        """
        store = CNContactStore.alloc().init()
        request = CNSaveRequest.alloc().init()
        request.addContact_toContainerWithIdentifier_(self.contact, None)

        error = objc.nil
        success, error = store.executeSaveRequest_error_(request, None)
        if not success:
            print(f"Failed to save contact: {error}")
        else:
            print("Contact saved successfully.")

    def __str__(self):
        properties = [(attr, getattr(self.contact, attr)())
                      for attr in dir(self.contact) if not attr.startswith("__")]
        properties_str = "\n".join(
            f"{name}: {value}" for name, value in properties)
        return properties_str

    def __repr__(self):
        return f"Contact({self.contact.givenName()}, {self.contact.familyName()}, {self.contact.emailAddresses()[0].value()}, {self.contact.phoneNumbers()[0].value()})"


def parse_args():
    # i'll add a command line tool later
    # for now this is just a stub / dummy data
    # similar to the way it'll look from docopt
    return {
        'first': 'John99',
        'last': 'Doe99',
        'email': 'john99.doe99@example.com',
        'phone': '555-9999',
        'title': 'Software Engineer',
        'company': 'Tech Innovations Inc.',
        'website': 'https://www.tech-innovations.com'
    }


if __name__ == "__main__":
    data = parse_args()
    first, last, email, phone, title, company, website = data.values()

    contact = Contact(
        first,
        last,
        email,
        phone,
        title,
        company,
        website,
        note='foobar'
    )
    contact.save()
