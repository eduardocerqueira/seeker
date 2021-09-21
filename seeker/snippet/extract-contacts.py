#date: 2021-09-21T17:08:54Z
#url: https://api.github.com/gists/f2f04006b64437fe208dad93a01579d9
#owner: https://api.github.com/users/esqew

# extract-contents.py
# by Sean F Quinn <sean@sfq.xyz>
# generates a contacts.vcf in the cwd based on the contacts XML output present in an Outlook for Mac mailbox export file (`.olm`)
#
# instructions:
#   1. open *.olm file in text editor
#   2. manually extract all <contacts></contacts> nodes present into separate *.xml files
#   3. point minidom.parse() on line 15 to your generated xml file
#
# requires vobject (pip install vobject)

from xml.dom import minidom
from pprint import pprint
import vobject

xmldoc = minidom.parse('./path/to/contacts.xml')
contacts = xmldoc.getElementsByTagName('contact')
cards = []

for contact in contacts:
    card = vobject.vCard()
    firstname = ''
    lastname = ''
    for attribute in contact.childNodes:
        if attribute.nodeName == "OPFContactCopyDisplayName":
            card.add('fn').value = attribute.firstChild.nodeValue
        elif attribute.nodeName == "OPFContactCopyFirstName":
            firstname = attribute.firstChild.nodeValue
        elif attribute.nodeName == "OPFContactCopyLastName":
            lastname = attribute.firstChild.nodeValue
        elif attribute.nodeName == "OPFContactCopyHomePhone" or attribute.nodeName == "OPFContactCopyHomePhone2":
            card.add('tel').value = attribute.firstChild.nodeValue
            card.tel.type_param = 'voice,home'
        elif attribute.nodeName == "OPFContactCopyHomeFax":
            card.add('tel').value = attribute.firstChild.nodeValue
            card.tel.type_param = 'fax,home'
        elif (attribute.nodeName == "OPFContactCopyNotesPlain" or attribute.nodeName == "OPFContactcopyNotes") and attribute.firstChild.nodeValue.strip() != '':
            card.add('note').value = attribute.firstChild.nodeValue
        elif attribute.nodeName == "OPFContactCopyCellPhone":
            card.add('tel').value = attribute.firstChild.nodeValue
            card.tel.type_param = 'voice,cell'
        elif attribute.nodeName == "OPFContactCopyBirthday":
            card.add('bday').value = attribute.firstChild.nodeValue
        elif attribute.nodeName == "OPFContactCopyEmailAddressList":
            for emails in attribute.childNodes:
                card.add('email').value = emails.getAttribute('OPFContactEmailAddressAddress')
        elif attribute.nodeName == "OPFContactCopyBusinessCompany":
            card.add('org').value = [attribute.firstChild.nodeValue]

        if attribute.nodeName == "OPFContactCopyLastName" or attribute.nodeName == "OPFContactCopyFirstName" and (firstname != "" and lastname != ""):
            card.add('n').value = vobject.vcard.Name( family=lastname, given=firstname)

    cards.append(card)

f = open('contacts.vcf', 'w')

for card in cards:
    if not card.fn:
        print(card.serialize())
    f.write(card.serialize())

f.close()