//date: 2022-12-27T17:00:24Z
//url: https://api.github.com/gists/296eef6af6c9b1f9e78b6299a24956fc
//owner: https://api.github.com/users/albuquerque53

package integration

import (
	"database/sql"
	"net/http/httptest"
	"usersapi/internal/infra/db"

	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"
)

type Suite struct {
	// obrigatório para funcionamento da suite
	suite.Suite

	// conexão com o DB que usaremos nos testes
	DBConn *sql.DB

	// servidor de teste necessário para chamarmos as rotas
	Srv *httptest.Server

	// instância de migration
	Migration *db.Migration
}

// SetupSuite será chamado sempre que formos rodar um arquivo de teste
// ele deve montar todo o ambiente necessário para a execução do mesmo.
func (s *Suite) SetupSuite() {
	var err error

	s.DBConn = db.ConectToDatabase()

	// intanciando as migrations (não rodando)
	s.Migration, err = db.RunMigration(s.DBConn, "../infra/db/migrations")
	require.NoError(s.T(), err)
}

// TearDownSuite será chamado sempre todos os testes do arquivo terminarem
// a execução, ele deve derrubar todo o ambiente montado pelo SetupSuite.
func (s *Suite) TearDownSuite() {
	s.DBConn.Close()
}
