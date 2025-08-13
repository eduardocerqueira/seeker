//date: 2025-08-13T17:01:12Z
//url: https://api.github.com/gists/32055e990c9ddbcae09782d05460ca04
//owner: https://api.github.com/users/julioscheidtsigma

// go get github.com/jackc/pgx/v5
// go get github.com/aws/aws-sdk-go-v2/aws
// go get github.com/aws/aws-sdk-go-v2/config
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/jackc/pgx/v5"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	cognito "github.com/aws/aws-sdk-go-v2/service/cognitoidentityprovider"
	"github.com/aws/aws-sdk-go-v2/service/cognitoidentityprovider/types"
)

type DatabaseUser struct {
	LoweredEmail string
	FoundEmails  []string
	FoundIDs     []string
}

type CognitoUser struct {
	Username      string
	Email         string
	UserStatus    types.UserStatusType
	HasExactEmail bool
}

func disableSecondaryDatabaseUsers(ctx context.Context, conn *pgx.Conn, loweredEmail string, dryRun bool) error {
	log.Printf("Disabling secondary database users with email %s\n", loweredEmail)
	if dryRun {
		return nil
	}
	// it will disable all users with the same email but with different case
	query := `
		update users
		set disabled = $1
		where lower(email) = $2 and email <> $2
	`
	_, err := conn.Exec(ctx, query, true, loweredEmail)
	return err
}

func updateDatabaseUserEmail(ctx context.Context, conn *pgx.Conn, id, loweredEmail string, dryRun bool) error {
	log.Printf("Updating database user %s with email %s\n", id, loweredEmail)
	if dryRun {
		return nil
	}
	query := `
		update users
		set email = $1
		where id = $2
	`
	_, err := conn.Exec(ctx, query, loweredEmail, id)
	return err
}

func findCognitoUsersByEmail(ctx context.Context, cognitoClient *cognito.Client, email, userPoolID string) ([]CognitoUser, error) {
	baseInput := &cognito.ListUsersInput{
		UserPoolId: aws.String(userPoolID),
		Limit:      aws.Int32(10),
	}

	baseInput.Filter = aws.String(fmt.Sprintf("email = \"%s\"", email))
	output, err := cognitoClient.ListUsers(ctx, baseInput)
	if err != nil {
		return []CognitoUser{}, err
	}

	resultUsers := make([]CognitoUser, 0)

	for _, user := range output.Users {
		username := aws.ToString(user.Username)
		hasExactEmail := false
		currentEmail := ""

		for _, attr := range user.Attributes {
			if aws.ToString(attr.Name) == "email" {
				currentEmail = aws.ToString(attr.Value)
				if currentEmail == email {
					hasExactEmail = true
				}
			}
		}

		resultUsers = append(resultUsers, CognitoUser{
			Username:      username,
			Email:         currentEmail,
			UserStatus:    user.UserStatus,
			HasExactEmail: hasExactEmail,
		})
	}

	return resultUsers, nil
}

func deleteCognitoUser(ctx context.Context, client *cognito.Client, username, userPoolID string, dryRun bool) error {
	log.Printf("Deleting Cognito user %s from user pool %s\n", username, userPoolID)
	if dryRun {
		return nil
	}
	_, err := client.AdminDeleteUser(ctx, &cognito.AdminDeleteUserInput{
		UserPoolId: aws.String(userPoolID),
		Username:   aws.String(username),
	})
	return err
}

func updateCognitoUserEmail(ctx context.Context, client *cognito.Client, user CognitoUser, loweredEmail, userPoolID string, dryRun bool) error {
	log.Printf("Updating Cognito user %s with email %s in user pool %s\n", user.Username, loweredEmail, userPoolID)

	if user.HasExactEmail {
		log.Printf("Cognito user %s already has the correct email %s\n", user.Username, loweredEmail)
		return nil
	}

	if dryRun {
		return nil
	}

	input := &cognito.AdminUpdateUserAttributesInput{
		UserPoolId: aws.String(userPoolID),
		Username:   aws.String(user.Username),
		UserAttributes: []types.AttributeType{
			{
				Name:  aws.String("email"),
				Value: aws.String(loweredEmail),
			},
			{
				Name:  aws.String("email_verified"),
				Value: aws.String("true"),
			},
		},
	}
	_, err := client.AdminUpdateUserAttributes(ctx, input)
	return err
}

func processCognitoUsers(ctx context.Context, cognitoClient *cognito.Client, correctEmail, userPoolID string, dryRun bool) error {
	// try to find at first with the lowered email
	cognitoUsers, err := findCognitoUsersByEmail(ctx, cognitoClient, correctEmail, userPoolID)
	if err != nil {
		return err
	}

	if len(cognitoUsers) == 0 {
		// if we did not find any user, we can just return
		log.Printf("No cognito users found with email %s\n", correctEmail)
		return nil
	}

	if len(cognitoUsers) == 1 {
		// if we found only one user, we can update it with the correct email
		cognitoUser := cognitoUsers[0]
		return updateCognitoUserEmail(ctx, cognitoClient, cognitoUser, correctEmail, userPoolID, dryRun)
	}

	// if we found more than one user, we'll keep the one with the correct email
	sort.SliceStable(cognitoUsers, func(i, j int) bool {
		return cognitoUsers[i].HasExactEmail && !cognitoUsers[j].HasExactEmail
	})

	// delete all but the first one
	for _, cognitoUser := range cognitoUsers[1:] {
		if err := deleteCognitoUser(ctx, cognitoClient, cognitoUser.Username, userPoolID, dryRun); err != nil {
			return err
		}
	}

	cognitoUser := cognitoUsers[0]
	if err := updateCognitoUserEmail(ctx, cognitoClient, cognitoUser, correctEmail, userPoolID, dryRun); err != nil {
		return err
	}

	return nil
}

func main() {
	dryRun := false
	if strings.ToLower(os.Getenv("DRY_RUN")) == "true" {
		log.Println("Running in dry run mode, no changes will be made")
		dryRun = true
	}

	UserPoolID := os.Getenv("USER_POOL_ID")
	if UserPoolID == "" {
		// UserPoolID = "us-east-1_AAAAAAAAA"
	}

	ConnString := os.Getenv("DATABASE_URL")
	if ConnString == "" {
		// ConnString = "postgres: "**********":<PASSWORD>@<HOSTNAME>:<PORT>/<DB_NAME>"
	}

	ctx := context.Background()

	cfg, err := config.LoadDefaultConfig(ctx)
	if err != nil {
		log.Fatalf("unable to load AWS config: %v", err)
	}
	cognitoClient := cognito.NewFromConfig(cfg)

	conn, err := pgx.Connect(ctx, ConnString)
	if err != nil {
		log.Fatalf("Unable to connect to database: %v\n", err)
	}
	defer conn.Close(ctx)

	// fetch all users with uppercase letters in their email, not coming from SSO, grouped by their lowercase email
	query := `
		select
			lowered_email,
			array_agg(distinct email) filter (where email is not null) as found_emails,
			array_agg(distinct id) filter (where id is not null) as found_ids
		from (
			select
				lower(usr1.email) as lowered_email,
				unnest(array[usr1.email, usr2.email]) as email,
				unnest(array[usr1.id, usr2.id]) as id
			from users usr1
				left join lateral (
					select id, email, lower(email) as lowered_email, organization_id, notes from users
					where (notes is null || notes is not null and notes <> 'SSO')
				) as usr2 on lower(usr1.email) = usr2.lowered_email and usr1.id <> usr2.id and usr1.organization_id = usr2.organization_id
			where
				usr1.email <> lower(usr1.email)
				and (usr1.notes is null || usr1.notes is not null and usr1.notes <> 'SSO')
		) sub_1
		group by 1
	`

	rows, err := conn.Query(ctx, query)
	if err != nil {
		log.Fatalf("Query failed: %v\n", err)
	}
	defer rows.Close()

	databaseUsers := make([]DatabaseUser, 0)
	for rows.Next() {
		var u DatabaseUser
		err := rows.Scan(&u.LoweredEmail, &u.FoundEmails, &u.FoundIDs)
		if err != nil {
			log.Fatalf("Row scan failed: %v\n", err)
		}
		databaseUsers = append(databaseUsers, u)
	}

	for _, databaseUser := range databaseUsers {
		// no duplicate users on the database, just found one with the incorrect email
		if len(databaseUser.FoundEmails) == 1 && len(databaseUser.FoundIDs) == 1 {
			if err = updateDatabaseUserEmail(ctx, conn, databaseUser.FoundIDs[0], databaseUser.LoweredEmail, dryRun); err != nil {
				log.Fatalf("Update database failed: %v\n", err)
			}
		} else {
			// disable the other users on the database
			if err = disableSecondaryDatabaseUsers(ctx, conn, databaseUser.LoweredEmail, dryRun); err != nil {
				log.Fatalf("Set disabled database user failed: %v\n", err)
			}
		}

		if err = processCognitoUsers(ctx, cognitoClient, databaseUser.LoweredEmail, UserPoolID, dryRun); err != nil {
			log.Fatalf("Cognito user processing failed: %v\n", err)
		}
	}
}
