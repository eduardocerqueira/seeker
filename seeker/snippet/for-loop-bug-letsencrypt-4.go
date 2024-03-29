//date: 2024-03-29T17:09:08Z
//url: https://api.github.com/gists/066e5b392f1d287a17acee3f26a49778
//owner: https://api.github.com/users/matteo-pampana

func authz2ModelMapToPB(m map[string]authz2Model) (*sapb.Authorizations, error) {
    // ...
    for k, v := range m {
        authzPB, err := modelToAuthzPB(&v)
    }
    // ...
}