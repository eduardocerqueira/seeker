//date: 2023-03-14T17:06:46Z
//url: https://api.github.com/gists/4e5303396df3d91fd18e05c327b597ae
//owner: https://api.github.com/users/croatiangrn

func importContactsViaIntegration(clientID int64, list model.List, srcContacts []RequestContactType, options RequestTypeOptions, runner sq.StdSql) ([]int64, error) {
	usedNumbers := make(map[string]int64)

	batchID, _ := util.NewUUID()

	multiErrors := util.MultiErrors{}
	internalErrors := util.MultiErrors{}

	filename := fmt.Sprintf("assets/sync/%s.json", batchID)
	f, err := os.Create(filename)
	if err != nil {
		return nil, &util.InternalErrorType{Msg: errors.WithStack(err).Error()}
	}
	defer f.Close()

	chunkSize := 2000

	insertIDs := make([]int64, 0, chunkSize)

	var insertContactsQueryBuilder strings.Builder
	insertContactsQueryValuesPlaceholder := strings.Repeat("?, ", 35)
	insertContactsQueryValuesPlaceholder = strings.TrimSuffix(insertContactsQueryValuesPlaceholder, ", ")

	contactsPayloadInsert := make([]model.Contact, 0, chunkSize)
	insertContactsQueryParams := make([]interface{}, 0, chunkSize*35)

	updateIDs := make([]int64, 0, chunkSize)

	var updateContactsQueryBuilder strings.Builder
	updateContactsQueryValuesPlaceholder := strings.Repeat("?, ", 36)
	updateContactsQueryValuesPlaceholder = strings.TrimSuffix(updateContactsQueryValuesPlaceholder, ", ")

	contactsPayloadUpdate := make([]model.Contact, 0, chunkSize)
	updateContactsQueryParams := make([]interface{}, 0, chunkSize*35)

	/**
	These columns should maybe be dropped from the table ?
	phonecode, phonenumber, phonecode, phonenumber, province, altphonenumber, scoredata, numberscount, mobilecount,
	landlinecount,
	*/

	insertPartOfQuery := `INSERT INTO 
    				contacts (clientid, vendorcontactid, title, firstname, middlename, lastname,
                      address, city, state, postalcode, country, mailingaddress, mailingcity, mailingstate,
                      mailingpostalcode, gender, dateofbirth, email, securityphrase, comments,
                      customfields, addressurls, score, recalcscore, status, numbertype, lastcallid,
                      currentagentid, datelasttouched, dialedcount, federaldnc, listscount, 
                      campaignscount, dateadded, datemodified)
					VALUES `
	updatePartOfQuery := `INSERT INTO 
    				contacts (id, clientid, vendorcontactid, title, firstname, middlename, lastname,
                      address, city, state, postalcode, country, mailingaddress, mailingcity, mailingstate,
                      mailingpostalcode, gender, dateofbirth, email, securityphrase, comments,
                      customfields, addressurls, score, recalcscore, status, numbertype, lastcallid,
                      currentagentid, datelasttouched, dialedcount, federaldnc, listscount, 
                      campaignscount, dateadded, datemodified)
					VALUES `

	insertContactsQueryBuilder.WriteString(insertPartOfQuery)
	updateContactsQueryBuilder.WriteString(updatePartOfQuery)

	insertInBulkCounter := 0
	updateInBulkCounter := 0

	updateContactPhoneTypesQuery := `UPDATE contacts SET mobilecount = ?, landlinecount = ?, dnccount = ?, numberscount = ? WHERE id = ?`

	for i := range srcContacts {
		log.Printf("Processing contact #%d", i)

		fmt.Fprintf(f, "Row %d: source contact: %+v\n", i, srcContacts[i])

		contact := model.Contact{
			ClientId: clientID,
		}

		numbers := []string{
			srcContacts[i].PhoneNumber1,
			srcContacts[i].PhoneNumber2,
			srcContacts[i].PhoneNumber3,
			srcContacts[i].PhoneNumber4,
			srcContacts[i].PhoneNumber5,
			srcContacts[i].PhoneNumber6,
			srcContacts[i].PhoneNumber7,
			srcContacts[i].PhoneNumber8,
			srcContacts[i].PhoneNumber9,
			srcContacts[i].PhoneNumber10,
		}

		if (srcContacts[i].PhoneNumber1 == "") && (srcContacts[i].PhoneNumber != "") {
			numbers[0] = srcContacts[i].PhoneNumber
		}

		if (srcContacts[i].PhoneNumber2 == "") && (srcContacts[i].AltPhoneNumber != "") {
			numbers[1] = srcContacts[i].AltPhoneNumber
		}

		newNumbersCounter := 0
		var numbersSlice []string
		phoneNumbers := make([]model.PhoneNumber, 0, len(numbers))

		for j, number := range numbers {
			num := util.FixPhoneNumber(number)
			if len(num) == 10 {
				numbers[j] = num
				if id, ok := usedNumbers[num]; ok {
					log.Printf("Row %d: skipping number %s as used for contact #%d", i, num, id)
					fmt.Fprintf(f, "Row %d: skipping number %s as used for contact #%d\n", i, num, id)
					numbers[j] = ""
				} else {
					newNumbersCounter++
					numbersSlice = append(numbersSlice, num)
					phoneNumbers = append(phoneNumbers, model.PhoneNumber{
						NumberID:    int64(newNumbersCounter),
						PhoneNumber: number,
					})
				}
			} else {
				numbers[j] = ""
			}
		}

		numbers = numbersSlice

		if newNumbersCounter == 0 {
			log.Print("All contact's phone numbers are used by another contacts from the batch, skipping")
			multiErrors = append(multiErrors, errors.Errorf("Contact %d: all contact's phone numbers are used by another contacts from the batch, skipping", i))
			continue
		}

		contacts, err := model.ContactsByNumber(clientID, runner, numbers)
		if err != nil {
			log.Printf("%v - %v", err, srcContacts[i])
			internalErrors = append(internalErrors, errors.Errorf("Contact %d: error loading contacts by numbers: %v", i, err))
			continue
		}

		/**
		 * This part of code DOESN'T update phone numbers
		 */
		if (contacts != nil) && len(contacts) > 0 {
			for j, contact := range contacts {
				log.Printf("Adding existing contact %d: %v", contact.Id, srcContacts[i])
				fmt.Fprintf(f, "Row %d: adding existing contact %d to the list\n", i, contact.Id)
				contacts[j].WithLists(runner)
				contacts[j].WithPhoneNumbers(runner)
				contacts[j].Lists = append(contacts[j].Lists, list)
				if srcContacts[i].IgnoreVendorID == 0 {
					if srcContacts[i].GetVendorContactId() == contacts[j].VendorContactId {
						updateContact(&contacts[j], srcContacts[i])
					} else if strings.EqualFold(contact.FirstName, srcContacts[i].FirstName) && strings.EqualFold(contact.LastName, srcContacts[i].LastName) {
						hasPhoneValidationErrors := false
						if len(srcContacts[i].PhoneNumber) > 0 {
							srcContacts[i].PhoneNumber = util.RemoveExtraSpacesFromString(srcContacts[i].PhoneNumber)
							srcContacts[i].PhoneNumber = util.StripNonNumericCharacters(srcContacts[i].PhoneNumber)

							if !util.IsPhoneNumber(srcContacts[i].PhoneNumber) {
								hasPhoneValidationErrors = true
								multiErrors = append(multiErrors, errors.Errorf("Contact %d: %q is not valid phone number. Phone number must be numeric value that is 10 digits long", j, srcContacts[i].PhoneNumber))
							}
						}

						if len(srcContacts[i].PhoneNumber1) > 0 {
							srcContacts[i].PhoneNumber1 = util.RemoveExtraSpacesFromString(srcContacts[i].PhoneNumber1)
							srcContacts[i].PhoneNumber1 = util.StripNonNumericCharacters(srcContacts[i].PhoneNumber1)

							if len(srcContacts[i].PhoneNumber1) > 0 && !util.IsPhoneNumber(srcContacts[i].PhoneNumber1) {
								hasPhoneValidationErrors = true
								multiErrors = append(multiErrors, errors.Errorf("Contact %d: %q is not valid phone number. Phone number must be numeric value that is 10 digits long", j, srcContacts[i].PhoneNumber))
							}
						}

						if !hasPhoneValidationErrors {
							for _, number := range contacts[j].PhoneNumbers {
								if srcContacts[i].PhoneNumber1 == number.PhoneNumber || srcContacts[i].PhoneNumber == number.PhoneNumber {
									updateContact(&contacts[j], srcContacts[i])
									break
								}
							}
						}
					} else {
						multiErrors = append(multiErrors, errors.Errorf("Contact %d: Fields %q, %q and %q have different values from contact with the ID of %d and therefore this contact can't be updated", i, "vendorcontactid", "firstname", "lastname", contact.Id))
					}
				} else {
					matches := 0
					for _, num1 := range contacts[j].PhoneNumbers {
						for _, num2 := range numbers {
							if num1.PhoneNumber == num2 && num2 != "" {
								matches++
							}
						}
					}
					if matches >= 2 || matches == len(contacts[j].PhoneNumbers) {
						if srcContacts[i].VendorContactId != "" {
							contacts[j].VendorContactId = srcContacts[i].GetVendorContactId()
						}
						updateContact(&contacts[j], srcContacts[i])
					}
				}
			}
		} else {
			contact = model.Contact{}
			contact.ClientId = clientID
			contact.Lists = []model.List{list}
			contact.Title = strings.TrimSpace(srcContacts[i].Title)
			contact.FirstName = strings.TrimSpace(srcContacts[i].FirstName)
			contact.MiddleName = strings.TrimSpace(srcContacts[i].MiddleName)
			contact.LastName = strings.TrimSpace(srcContacts[i].LastName)
			contact.Gender = strings.TrimSpace(srcContacts[i].Gender)
			contact.DateOfBirth = srcContacts[i].DateOfBirth
			contact.VendorContactId = strings.TrimSpace(srcContacts[i].GetVendorContactId())

			contact.Address = strings.TrimSpace(srcContacts[i].Address)
			contact.City = strings.TrimSpace(srcContacts[i].City)
			contact.State = strings.TrimSpace(srcContacts[i].State)
			contact.PostalCode = strings.TrimSpace(srcContacts[i].PostalCode)

			contact.MailingAddress = strings.TrimSpace(srcContacts[i].MailingAddress)
			contact.MailingCity = strings.TrimSpace(srcContacts[i].MailingCity)
			contact.MailingState = strings.TrimSpace(srcContacts[i].MailingState)
			contact.MailingPostalCode = strings.TrimSpace(srcContacts[i].MailingPostalCode)

			contact.Country = strings.TrimSpace(srcContacts[i].Country)
			contact.Email = strings.TrimSpace(srcContacts[i].Email)
			contact.Comments = strings.TrimSpace(srcContacts[i].Comments)
			contact.CustomFields = srcContacts[i].CustomFields
			contact.Status = "active"
			contact.PhoneNumbers = phoneNumbers
			contacts = []model.Contact{contact}
		}

		type PhoneCheckedInfo struct {
			Type *string
			DNC  *bool
		}

		phonesCheckedInfo := make(map[string]PhoneCheckedInfo)
		if options.IsAnyChecked() {
			checkPhonesInfo := func(numbers []string) {
				if typeInfoList, err := util.GetPhoneNumberTypes(numbers); err != nil {
					log.Printf("[importContacts] Can't check phone type in reicore API: %v", err)
				} else {
					for _, typeInfo := range typeInfoList {
						if info, ok := phonesCheckedInfo[typeInfo.Number]; ok {
							info.Type = &typeInfo.Type
							phonesCheckedInfo[typeInfo.Number] = info
						} else {
							phonesCheckedInfo[typeInfo.Number] = PhoneCheckedInfo{
								Type: &typeInfo.Type,
							}
						}
					}
				}

				if dncInfoList, err := util.GetPhoneNumberDNC(numbers); err != nil {
					log.Printf("[importContacts] Can't check phone type in reicore API: %v", err)
				} else {
					for _, dncInfo := range dncInfoList {
						if info, ok := phonesCheckedInfo[dncInfo.Number]; ok {
							info.DNC = &dncInfo.DNC
							phonesCheckedInfo[dncInfo.Number] = info
						} else {
							phonesCheckedInfo[dncInfo.Number] = PhoneCheckedInfo{
								DNC: &dncInfo.DNC,
							}
						}
					}
				}
			}
			batchSize := 100
			phonesList := make([]string, 0, batchSize)
			for _, contact := range contacts {
				for _, phoneNumber := range contact.PhoneNumbers {
					phonesList = append(phonesList, phoneNumber.PhoneNumber)
					if len(phonesList) == batchSize {
						checkPhonesInfo(phonesList)
						phonesList = make([]string, 0, batchSize)
					}
				}
			}
			if len(phonesList) > 0 {
				checkPhonesInfo(phonesList)
			}
		}

		for _, contact := range contacts {
			if err := contact.ValidateWithRunner(model.ContactShouldCheckPhoneBelonging, runner); err != nil {
				log.Printf("Row %d: contact validation error: %v", i, err)
				multiErrors = append(multiErrors, errors.Errorf("Contact %d: contact validation error: %v", i, err))
				continue
			}
			if options.IsAnyChecked() {
				for _, phoneNumber := range contact.PhoneNumbers {
					info, ok := phonesCheckedInfo[phoneNumber.PhoneNumber]
					if !ok {
						errMsg := fmt.Sprintf(
							"Row %d: contact phone %s was removed, beacause can't check phone status in reicore API",
							i, phoneNumber.PhoneNumber,
						)
						log.Println(errMsg)
						fmt.Fprintln(f, errMsg)
						continue
					}
					if options.FederalDNC {
						if info.DNC == nil {
							errMsg := fmt.Sprintf(
								"Row %d: contact phone %s was removed, beacause can't check DNC in reicore API",
								i, phoneNumber.PhoneNumber,
							)
							log.Println(errMsg)
							fmt.Fprintln(f, errMsg)
							continue
						}
						if *info.DNC {
							errMsg := fmt.Sprintf(
								"Row %d: contact phone %s was removed by Federal DNC condition",
								i, phoneNumber.PhoneNumber,
							)
							log.Println(errMsg)
							fmt.Fprintln(f, errMsg)
							continue
						}
					}
					if options.SystemDNC {
						isDNC, _ := contact.IsCompanyDnc(runner, phoneNumber.PhoneNumber)
						if isDNC {
							errMsg := fmt.Sprintf(
								"Row %d: contact phone %s was removed by System DNC condition",
								i, phoneNumber.PhoneNumber,
							)
							log.Println(errMsg)
							fmt.Fprintln(f, errMsg)
							continue
						}
					}
					if options.Landlines || options.Mobiles || options.Unknown || options.WrongNumbers {
						if info.Type == nil {
							errMsg := fmt.Sprintf(
								"Row %d: contact phone %s was removed, beacause can't check phone type in reicore API",
								i, phoneNumber.PhoneNumber,
							)
							log.Println(errMsg)
							fmt.Fprintln(f, errMsg)
							continue
						}

						switch *info.Type {
						case "Mobile":
							if !options.Mobiles {
								errMsg := fmt.Sprintf(
									"Row %d: contact phone %s was removed by Mobile condition",
									i, phoneNumber.PhoneNumber,
								)
								log.Println(errMsg)
								fmt.Fprintln(f, errMsg)
								continue
							}
						case "Land Line":
							if !options.Landlines {
								errMsg := fmt.Sprintf(
									"Row %d: contact phone %s was removed by Landlines condition",
									i, phoneNumber.PhoneNumber,
								)
								log.Println(errMsg)
								fmt.Fprintln(f, errMsg)
								continue
							}
						default:
							if *info.Type != "" {
								if !options.Unknown && !options.WrongNumbers {
									errMsg := fmt.Sprintf(
										"Row %d: contact phone %s was removed by Unknown or Wrong Numbers condition",
										i, phoneNumber.PhoneNumber,
									)
									log.Println(errMsg)
									fmt.Fprintln(f, errMsg)
									continue
								}
							}
						}
					}
					contact.PhoneNumbers = append(contact.PhoneNumbers, phoneNumber)
				}

				if len(contact.PhoneNumbers) == 0 {
					errMsg := fmt.Sprintf(
						"Row %d: can't save contact: numbers list is empty",
						i,
					)
					log.Println(errMsg)
					fmt.Fprintln(f, errMsg)
					continue
				}
			}

			for _, num := range phoneNumbers {
				usedNumbers[num.PhoneNumber] = contact.Id
			}

			// Prepare bulk update
			if contact.Id > 0 {
				updateInBulkCounter++

				updateContactsQueryParams = append(updateContactsQueryParams, contact.Id, contact.ClientId, contact.VendorContactId, contact.Title, contact.FirstName, contact.MiddleName, contact.LastName, contact.Address, contact.City, contact.State, contact.PostalCode, contact.Country, contact.MailingAddress, contact.MailingCity, contact.MailingState, contact.MailingPostalCode, contact.Gender, contact.DateOfBirth, contact.Email, contact.SecurityPhrase, contact.Comments, contact.CustomFields, contact.AddressUrls, contact.Score, contact.RecalcScore, contact.Status, contact.NumberType, contact.LastCallId, contact.CurrentAgentId, contact.DateLastTouched, contact.DialedCount, contact.FederalDNC, contact.ListsCount, contact.CampaignsCount, contact.DateAdded, contact.DateModified)
				contactsPayloadUpdate = append(contactsPayloadUpdate, contact)

				if updateInBulkCounter == chunkSize {
					updateContactsQueryBuilder.WriteString(`(` + updateContactsQueryValuesPlaceholder + `) `)
					updateContactsQueryBuilder.WriteString(`ON DUPLICATE KEY UPDATE clientid = VALUES(clientid), vendorcontactid = VALUES(vendorcontactid), title = VALUES(title), firstname = VALUES(firstname), middlename = VALUES(middlename), lastname = VALUES(lastname), address = VALUES(address), city = VALUES(city), state = VALUES(state), postalcode = VALUES(postalcode), country = VALUES(country), mailingaddress = VALUES(mailingaddress), mailingcity = VALUES(mailingcity), mailingstate = VALUES(mailingstate), mailingpostalcode = VALUES(mailingpostalcode), gender = VALUES(gender), dateofbirth = VALUES(dateofbirth), email = VALUES(email), securityphrase = VALUES(securityphrase), comments = VALUES(comments), customfields = VALUES(customfields), addressurls = VALUES(addressurls), score = VALUES(score), recalcscore = VALUES(recalcscore), status = VALUES(status), numbertype = VALUES(numbertype), lastcallid = VALUES(lastcallid), currentagentid = VALUES(currentagentid), datelasttouched = VALUES(datelasttouched), dialedcount = VALUES(dialedcount), federaldnc = VALUES(federaldnc), listscount = VALUES(listscount), campaignscount = VALUES(campaignscount), dateadded = VALUES(dateadded), datemodified = VALUES(datemodified) RETURNING id`)

					queryRows, err := runner.Query(updateContactsQueryBuilder.String(), updateContactsQueryParams...)
					if err != nil {
						internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contacts: %s", err))
						continue
					}

					for queryRows.Next() {
						currID := int64(0)
						if err := queryRows.Scan(&currID); err != nil {
							internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contacts: %s", err))
							queryRows.Close()
							break
						}

						updateIDs = append(updateIDs, currID)
						contactsPayloadUpdate[len(updateIDs)-1].Id = currID
					}

					queryRows.Close()

					var phoneNumbersBulkQueryBuilder strings.Builder
					phoneNumbersBulkQueryBuilder.WriteString("INSERT INTO phonenumbers (clientid, contactid, numberid, phonenumber, numbertype, dnc, tested, reachable) VALUES ")
					var phoneNumbersQueryParams []interface{}

					for k := range contactsPayloadUpdate {
						numbersPlaceholders, numbersParams, phoneNumberTypes, err := contactsPayloadUpdate[k].SaveOnContactsImport(runner)
						if err != nil {
							internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers: %s", err))
							continue
						}

						phoneNumbersBulkQueryBuilder.WriteString(numbersPlaceholders)
						phoneNumbersQueryParams = append(phoneNumbersQueryParams, numbersParams...)

						if _, err := runner.Exec(updateContactPhoneTypesQuery, phoneNumberTypes[model.PhoneNumberTypeMobile], phoneNumberTypes[model.PhoneNumberTypeLandLine], phoneNumberTypes[model.PhoneNumberTypeDNC], phoneNumberTypes[model.PhoneNumbersCount], contactsPayloadUpdate[k].Id); err != nil {
							internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers count: %s", err))
							continue
						}
					}

					if len(phoneNumbersQueryParams) > 0 {
						query := strings.TrimSuffix(util.RemoveExtraSpacesFromString(phoneNumbersBulkQueryBuilder.String()), ",")
						query = query + "ON DUPLICATE KEY UPDATE phonenumber = VALUES(phonenumber), numbertype = VALUES(numbertype), dnc = VALUES(dnc), tested = VALUES(tested), reachable = VALUES(reachable)"

						if _, err := runner.Exec(query, phoneNumbersQueryParams...); err != nil {
							internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contact phone numbers: %s", err))
							continue
						} else {
							log.Printf("[phonenumbers] inserted in bulk")
						}
					}

					updateContactsQueryBuilder.Reset()
					updateContactsQueryBuilder.WriteString(updatePartOfQuery)
					updateInBulkCounter = 0
					updateIDs = []int64{}
					updateContactsQueryParams = []interface{}{}
				} else {
					updateContactsQueryBuilder.WriteString(`(` + updateContactsQueryValuesPlaceholder + `), `)
				}

				continue
			}

			// Prepare bulk insert
			insertInBulkCounter++

			insertContactsQueryParams = append(insertContactsQueryParams, contact.ClientId, contact.VendorContactId, contact.Title, contact.FirstName, contact.MiddleName, contact.LastName, contact.Address, contact.City, contact.State, contact.PostalCode, contact.Country, contact.MailingAddress, contact.MailingCity, contact.MailingState, contact.MailingPostalCode, contact.Gender, contact.DateOfBirth, contact.Email, contact.SecurityPhrase, contact.Comments, contact.CustomFields, contact.AddressUrls, contact.Score, contact.RecalcScore, contact.Status, contact.NumberType, contact.LastCallId, contact.CurrentAgentId, contact.DateLastTouched, contact.DialedCount, contact.FederalDNC, contact.ListsCount, contact.CampaignsCount, contact.DateAdded, contact.DateModified)
			contactsPayloadInsert = append(contactsPayloadInsert, contact)

			if insertInBulkCounter == chunkSize {
				insertContactsQueryBuilder.WriteString(`(` + insertContactsQueryValuesPlaceholder + `) RETURNING id`)

				queryRows, err := runner.Query(insertContactsQueryBuilder.String(), insertContactsQueryParams...)
				if err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error inserting multiple contacts: %s", err))
					continue
				}

				for queryRows.Next() {
					currID := int64(0)
					if err := queryRows.Scan(&currID); err != nil {
						internalErrors = append(internalErrors, errors.Errorf("Error inserting multiple contacts: %s", err))
						queryRows.Close()
						break
					}

					insertIDs = append(insertIDs, currID)
					contactsPayloadInsert[len(insertIDs)-1].Id = currID
				}

				queryRows.Close()

				var phoneNumbersBulkQueryBuilder strings.Builder
				phoneNumbersBulkQueryBuilder.WriteString("INSERT INTO phonenumbers (clientid, contactid, numberid, phonenumber, numbertype, dnc, tested, reachable) VALUES ")
				var phoneNumbersQueryParams []interface{}

				for k := range contactsPayloadInsert {
					numbersPlaceholders, numbersParams, phoneNumberTypes, err := contactsPayloadInsert[k].SaveOnContactsImport(runner)
					if err != nil {
						internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers: %s", err))
						continue
					}

					phoneNumbersBulkQueryBuilder.WriteString(numbersPlaceholders)
					phoneNumbersQueryParams = append(phoneNumbersQueryParams, numbersParams...)

					if _, err := runner.Exec(updateContactPhoneTypesQuery, phoneNumberTypes[model.PhoneNumberTypeMobile], phoneNumberTypes[model.PhoneNumberTypeLandLine], phoneNumberTypes[model.PhoneNumberTypeDNC], phoneNumberTypes[model.PhoneNumbersCount], contactsPayloadInsert[k].Id); err != nil {
						internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers count: %s", err))
						continue
					}
				}

				if len(phoneNumbersQueryParams) > 0 {
					query := strings.TrimSuffix(util.RemoveExtraSpacesFromString(phoneNumbersBulkQueryBuilder.String()), ",")
					query = query + "ON DUPLICATE KEY UPDATE phonenumber = VALUES(phonenumber), numbertype = VALUES(numbertype), dnc = VALUES(dnc), tested = VALUES(tested), reachable = VALUES(reachable)"

					if _, err := runner.Exec(query, phoneNumbersQueryParams...); err != nil {
						internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contact phone numbers: %s", err))
						continue
					} else {
						log.Printf("[phonenumbers] inserted in bulk")
					}
				}

				insertContactsQueryBuilder.Reset()
				insertContactsQueryBuilder.WriteString(insertPartOfQuery)
				insertInBulkCounter = 0
				insertContactsQueryParams = make([]interface{}, 0, chunkSize*35)
				contactsPayloadInsert = make([]model.Contact, 0, chunkSize)
				insertIDs = make([]int64, 0, chunkSize)
			} else {
				insertContactsQueryBuilder.WriteString(`(` + insertContactsQueryValuesPlaceholder + `), `)
			}
		}
	}

	if insertInBulkCounter > 0 {
		query := strings.TrimSuffix(util.RemoveExtraSpacesFromString(insertContactsQueryBuilder.String()), ",")
		query = query + " RETURNING id"

		queryRows, err := runner.Query(query, insertContactsQueryParams...)
		if err != nil {
			internalErrors = append(internalErrors, errors.Errorf("Error inserting multiple contacts: %s", err))
		} else {
			for queryRows.Next() {
				currID := int64(0)
				if err := queryRows.Scan(&currID); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error inserting multiple contacts: %s", err))
					queryRows.Close()
					break
				}

				insertIDs = append(insertIDs, currID)
				contactsPayloadInsert[len(insertIDs)-1].Id = currID
			}

			queryRows.Close()

			var phoneNumbersBulkQueryBuilder strings.Builder
			phoneNumbersBulkQueryBuilder.WriteString("INSERT INTO phonenumbers (clientid, contactid, numberid, phonenumber, numbertype, dnc, tested, reachable) VALUES ")
			var phoneNumbersQueryParams []interface{}

			for k := range contactsPayloadInsert {
				numbersPlaceholders, numbersParams, phoneNumberTypes, err := contactsPayloadInsert[k].SaveOnContactsImport(runner)
				if err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers: %s", err))
					continue
				}

				phoneNumbersBulkQueryBuilder.WriteString(numbersPlaceholders)
				phoneNumbersQueryParams = append(phoneNumbersQueryParams, numbersParams...)

				if _, err := runner.Exec(updateContactPhoneTypesQuery, phoneNumberTypes[model.PhoneNumberTypeMobile], phoneNumberTypes[model.PhoneNumberTypeLandLine], phoneNumberTypes[model.PhoneNumberTypeDNC], phoneNumberTypes[model.PhoneNumbersCount], contactsPayloadInsert[k].Id); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers count: %s", err))
					continue
				}
			}

			if len(phoneNumbersQueryParams) > 0 {
				queryPhoneNumbers := strings.TrimSuffix(util.RemoveExtraSpacesFromString(phoneNumbersBulkQueryBuilder.String()), ",")
				queryPhoneNumbers = queryPhoneNumbers + "ON DUPLICATE KEY UPDATE phonenumber = VALUES(phonenumber), numbertype = VALUES(numbertype), dnc = VALUES(dnc), tested = VALUES(tested), reachable = VALUES(reachable)"

				if _, err := runner.Exec(queryPhoneNumbers, phoneNumbersQueryParams...); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contact phone numbers: %s", err))
				} else {
					log.Printf("[phonenumbers] inserted in bulk")
				}
			}
		}
	}

	if updateInBulkCounter > 0 {
		query := strings.TrimSuffix(util.RemoveExtraSpacesFromString(updateContactsQueryBuilder.String()), ",")
		query = query + `ON DUPLICATE KEY UPDATE clientid = VALUES(clientid), vendorcontactid = VALUES(vendorcontactid), title = VALUES(title), firstname = VALUES(firstname), middlename = VALUES(middlename), lastname = VALUES(lastname), address = VALUES(address), city = VALUES(city), state = VALUES(state), postalcode = VALUES(postalcode), country = VALUES(country), mailingaddress = VALUES(mailingaddress), mailingcity = VALUES(mailingcity), mailingstate = VALUES(mailingstate), mailingpostalcode = VALUES(mailingpostalcode), gender = VALUES(gender), dateofbirth = VALUES(dateofbirth), email = VALUES(email), securityphrase = VALUES(securityphrase), comments = VALUES(comments), customfields = VALUES(customfields), addressurls = VALUES(addressurls), score = VALUES(score), recalcscore = VALUES(recalcscore), status = VALUES(status), numbertype = VALUES(numbertype), lastcallid = VALUES(lastcallid), currentagentid = VALUES(currentagentid), datelasttouched = VALUES(datelasttouched), dialedcount = VALUES(dialedcount), federaldnc = VALUES(federaldnc), listscount = VALUES(listscount), campaignscount = VALUES(campaignscount), dateadded = VALUES(dateadded), datemodified = VALUES(datemodified) RETURNING id`

		queryRows, err := runner.Query(query, updateContactsQueryParams...)
		if err != nil {
			internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contacts: %s", err))
		} else {
			for queryRows.Next() {
				currID := int64(0)
				if err := queryRows.Scan(&currID); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contacts: %s", err))
					queryRows.Close()
					break
				}

				updateIDs = append(updateIDs, currID)
				contactsPayloadUpdate[len(updateIDs)-1].Id = currID
			}

			queryRows.Close()

			var phoneNumbersBulkQueryBuilder strings.Builder
			phoneNumbersBulkQueryBuilder.WriteString("INSERT INTO phonenumbers (clientid, contactid, numberid, phonenumber, numbertype, dnc, tested, reachable) VALUES ")
			var phoneNumbersQueryParams []interface{}

			for k := range contactsPayloadUpdate {
				numbersPlaceholders, numbersParams, phoneNumberTypes, err := contactsPayloadUpdate[k].SaveOnContactsImport(runner)
				if err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers: %s", err))
					continue
				}

				phoneNumbersBulkQueryBuilder.WriteString(numbersPlaceholders)
				phoneNumbersQueryParams = append(phoneNumbersQueryParams, numbersParams...)

				if _, err := runner.Exec(updateContactPhoneTypesQuery, phoneNumberTypes[model.PhoneNumberTypeMobile], phoneNumberTypes[model.PhoneNumberTypeLandLine], phoneNumberTypes[model.PhoneNumberTypeDNC], phoneNumberTypes[model.PhoneNumbersCount], contactsPayloadUpdate[k].Id); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error saving contacts phone numbers count: %s", err))
					continue
				}
			}

			if len(phoneNumbersQueryParams) > 0 {
				queryPhoneNumbers := strings.TrimSuffix(util.RemoveExtraSpacesFromString(phoneNumbersBulkQueryBuilder.String()), ",")
				queryPhoneNumbers = queryPhoneNumbers + "ON DUPLICATE KEY UPDATE phonenumber = VALUES(phonenumber), numbertype = VALUES(numbertype), dnc = VALUES(dnc), tested = VALUES(tested), reachable = VALUES(reachable)"

				if _, err := runner.Exec(queryPhoneNumbers, phoneNumbersQueryParams...); err != nil {
					internalErrors = append(internalErrors, errors.Errorf("Error updating multiple contact phone numbers: %s", err))
				} else {
					log.Printf("[phonenumbers] inserted in bulk")
				}
			}
		}
	}

	var ids []int64
	ids = append(ids, updateIDs...)
	ids = append(ids, insertIDs...)

	if len(internalErrors.Error()) > 0 {
		return ids, &util.InternalErrorType{Msg: internalErrors.Error()}
	}

	if len(multiErrors.Error()) > 0 {
		return ids, multiErrors
	}

	return ids, nil
}
