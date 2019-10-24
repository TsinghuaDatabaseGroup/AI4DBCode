package main

//存储现有的实例用于测试
type MysqlInst struct {
	InstId     int64  `json:inst_id`
	InstanceId string `json:instance_id`
	ClusterId  int64  `json:cluster_id`
	Host       string `json:host`
	Port       int64  `json:port`
	User       string `json:user`
	Password   string `json:password`
	MaxMem     int64  `json:max_mem`
	MaxDisk    int64  `json:max_disk`
	Tables     int64  `json:tables`
	TableSize  int64  `json:table_size`
}

func (m *MysqlInst) Insert(app *TuneServer) (int64, error) {
	sql := "insert into tb_mysql_inst(instance_id,cluster_id,host,port,user,password,max_mem,max_disk,tables,table_size) values('?',?,'?',?,'?','?',?,?,?,?)"
	rst, _ := app.conn.Exec(sql, m.InstanceId, m.ClusterId, m.Host, m.Port, m.User, m.Password, m.MaxMem, m.MaxDisk, m.Tables, m.TableSize)
	return rst.LastInsertId()
}
