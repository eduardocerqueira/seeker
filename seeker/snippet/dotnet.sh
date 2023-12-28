#date: 2023-12-28T16:41:15Z
#url: https://api.github.com/gists/5d2d2a5d2be9cbd9de96e2a8241ef033
#owner: https://api.github.com/users/benitoanagua

dotnet aspnet-codegenerator controller -name SurveyController -async -api -m Survey -dc SurveyContext -outDir Controllers -dbProvider postgres
dotnet ef migrations add InitialCreate
dotnet ef database update