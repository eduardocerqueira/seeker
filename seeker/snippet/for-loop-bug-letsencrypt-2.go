//date: 2024-03-29T17:06:42Z
//url: https://api.github.com/gists/e3a8a9507dec3c55133f35b4ffa6cbf7
//owner: https://api.github.com/users/matteo-pampana

func modelToAuthzPB(v *authzModel) (*corepb.Authorization, error) {
  //...
  pb := &corepb.Authorization{
    Id: &id,
    Status: &status,
    Identifier: &v.IdentifierValue,
    RegistrationID: &v.RegistrationID,
    Expires: &expires,
  }
  return pb, nil
}