//date: 2024-03-29T17:08:09Z
//url: https://api.github.com/gists/12d41017c22440a406b9f7564c267326
//owner: https://api.github.com/users/matteo-pampana

func authz2ModelMapToPB(m map[string]authz2Model) (*sapb.Authorizations, error) {
    // ...
    for k, v := range m {
        vCopy := v
        authzPB, err := modelToAuthzPB(&vCopy)
    }
    // ...
}