//date: 2022-10-31T17:28:20Z
//url: https://api.github.com/gists/48830d7950dc5463ebdd805ad7977ae5
//owner: https://api.github.com/users/vmesel

func (adapter *Postgres) QueryCtx(ctx context.Context, SQL string, params ...interface{}) (sc adapters.Scanner) {
	// use the db_name that was set on request to avoid runtime collisions
	db, err := getDBFromCtx(ctx)
	if err != nil {
		log.Errorln(err)
		return &scanner.PrestScanner{Error: err}
	}
	p, err := Prepare(db, SQL)
	if err != nil {
		log.Errorln(err)
		return &scanner.PrestScanner{Error: err}
	}
	var jsonData []byte
	rows, err := p.QueryContext(ctx, params...)
	if err != nil {
		log.Fatal(err)
	}
	var data []map[string]interface{}
	for rows.Next() {
		cols, err := rows.Columns()
		if err != nil {
			log.Fatal(err)
		}
		columns := make([]interface{}, len(cols))
		columnPointers := make([]interface{}, len(cols))
		for i := range columns {
			columnPointers[i] = &columns[i]
		}
		if err := rows.Scan(columnPointers...); err != nil {
			log.Fatal(err)
		}
		m := make(map[string]interface{})
		for i, colName := range cols {
			val := columnPointers[i].(*interface{})
			switch (*val).(type) {
				case []uint8:
					m[colName] = string((*val).([]byte))
				default:
					m[colName] = *val
			}
		}
		data = append(data, m)
	}
	jsonData, _ = json.Marshal(data)
	if len(jsonData) == 0 {
		jsonData = []byte("[]")
	}
	return &scanner.PrestScanner{
		Error:   err,
		Buff:    bytes.NewBuffer(jsonData),
		IsQuery: true,
	}
}