//date: 2021-11-08T17:18:22Z
//url: https://api.github.com/gists/9453807c4b34d1e69ff643fca57c52aa
//owner: https://api.github.com/users/ryan-holcombe

// FindAll returns all the users and their tickets
func (u UserTicketsService) FindAll(ctx context.Context) ([]UserWithTickets, error) {
    var results []UserWithTickets

    // retrieve all users
    users, err := u.userDAO.FindAll(ctx)
    if err != nil {
        return nil, err
    }

    // iterate over all users and find their tickets
    for _, user := range users {
        tickets, err := u.inventoryClient.FindUserTickets(user.ID)
        if err != nil {
            return nil, err
        }
        results = append(results, UserWithTickets{
            Name:    user.Name,
            Tickets: tickets,
        })
    }
    return results, nil
}