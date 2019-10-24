package main

import (
	"fmt"
	"reflect"
	"strings"

	. "git.code.oa.com/gocdb/base/public"
)

func getJsonTag(tName string) string {
	return "json:" + tName
}

//Query2Struct  model should use pointer and return slice is't pointer
//eg: app.Query2Struct("select * from tb_task_result",&TaskResult{} )
func (app *TuneServer) Query2Struct(sql string, model interface{}, values ...interface{}) ([]interface{}, error) {
	rst := []interface{}{}
	TLog.Infof("db query sql:%s value:+%v", sql, values)
	if rsp, err := app.conn.Query(sql, values...); err == nil {
		velems := reflect.ValueOf(model).Elem()
		telems := reflect.TypeOf(model).Elem()
		fieldMap := map[string]interface{}{}
		if cols, err := rsp.Columns(); err == nil {
			for i := 0; i < telems.NumField(); i++ {
				fieldMap[string(telems.Field(i).Tag)] = velems.Field(i).Addr().Interface()
			}
			scans := make([]interface{}, len(cols))
			for i, name := range cols {
				scans[i] = fieldMap[getJsonTag(name)]
			}
			for rsp.Next() {
				err := rsp.Scan(scans...)
				if err != nil {
					return rst, ErrSelectDb.AddErrMsg("row scan failed %+v", err)
				}
				rst = append(rst, velems.Interface())
			}
		} else {
			return rst, err
		}
	} else {
		return rst, err
	}
	return rst, nil
}

//Update
//eg: app.UpdateData(TB_TASK, "result_id", 3002, t)
func (app *TuneServer) DBUpdate(table, set, condition string, values ...interface{}) error {
	if !strings.Contains(condition, "=") {
		return ErrUpdateDb.AddErrMsg("not support update full table")
	}
	sql := "update %s set %s where %s"
	sql = fmt.Sprintf(sql, table, set, condition)
	TLog.Infof("db update sql:%s value:+%v", sql, values)
	if _, err := app.conn.Exec(sql, values...); err == nil {
		return nil
	} else {
		err := ErrUpdateDb.AddErrMsg("update failed +%v", err)
		if err != nil {
			TLog.Error(err)
		}
		return err
	}

}

func (app *TuneServer) DBInsert(sql string, values ...interface{}) (int64, error) {
	TLog.Infof("db insert sql:%s value:+%v", sql, values)
	if rst, err := app.conn.Exec(sql, values...); err == nil {
		id, err := rst.LastInsertId()
		return id, err
	} else {
		err = ErrInsertDb.AddErrMsg("insert failed +%v", err)
		if err != nil {
			TLog.Error(err)
		}
		return -1, err
	}
}

func (app *TuneServer) DBQuery(column, table, condition string, model interface{}, values ...interface{}) ([]interface{}, error) {
	sql := "select %s from %s where %s "
	sql = fmt.Sprintf(sql, column, table, condition)
	return app.Query2Struct(sql, model, values...)
}

//QueryByIndex return Pointer
//eg: t := &TaskResult{}
//eg: app.QueryByIndex(TB_TASK_RESULT, "result_id", 3002, t)
func (app *TuneServer) QueryByIndex(table, field string, id int64, model interface{}) error {
	sql := "%s = %d"
	sql = fmt.Sprintf(sql, field, id)
	rst, err := app.DBQuery("*", table, sql, model)
	if err == nil {
		if len(rst) != 1 {
			return ErrSelectDb.AddErrMsg(sql)
		}
		model = rst[0]
	}
	return err
}
