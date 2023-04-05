//date: 2023-04-05T17:03:12Z
//url: https://api.github.com/gists/4901873fef3500ffa8d23ea93182cb09
//owner: https://api.github.com/users/alexmwu


func TestAttestWithGCEAK(t *testing.T) {
	rwc := test.GetTPM(t)
	defer client.CheckedClose(t, rwc)
	ExternalTPM = rwc
	data, err := client.AKTemplateRSA().Encode()
	if err != nil {
		t.Fatalf("failed to encode AKTemplateRSA: %v", err)
	}
	idx := tpmutil.Handle(client.GceAKTemplateNVIndexRSA)
	if err := tpm2.NVDefineSpace(rwc, tpm2.HandlePlatform, idx,
		"", "", nil,
		tpm2.AttrPPWrite|tpm2.AttrPPRead|tpm2.AttrWriteDefine|tpm2.AttrOwnerRead|tpm2.AttrAuthRead|tpm2.AttrPlatformCreate|tpm2.AttrNoDA,
		uint16(len(data))); err != nil {
		t.Fatalf("NVDefineSpace failed: %v", err)
	}
	defer tpm2.NVUndefineSpace(rwc, "", tpm2.HandlePlatform, idx)
	err = tpm2.NVWrite(rwc, tpm2.HandlePlatform, idx, "", data, 0)
	if err != nil {
		t.Fatalf("failed to write NVIndex: %v", err)
	}
	k, err := client.GceAttestationKeyRSA(rwc)
	if err != nil {
		t.Fatalf("failed to open GCE AK RSA: %v", err)
	}
	defer k.Close()
	// Use key.
	t.Log(k)
}