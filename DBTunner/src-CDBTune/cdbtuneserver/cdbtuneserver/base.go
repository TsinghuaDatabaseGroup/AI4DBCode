package main

import (
	"bytes"
	"database/sql/driver"
	"fmt"
	"os/exec"
	"strings"

	. "git.code.oa.com/gocdb/base/public"
)

const (
	ER_UNKNOW int = 40001 + iota
	ER_DB_CONNECT
	ER_INTERNAL
	ER_CREATE_TASK
	ER_TUNE_FAILED
	ER_INVALID_PARAM

	TB_TASK        string = "tb_task"
	TB_TASK_RESULT string = "tb_task_result"
	TB_MYSQL_INST  string = "tb_mysql_inst"
	TB_MODELS      string = "tb_models"
)

var (
	ErrCreateTask = NewCDBError(ER_CREATE_TASK, "Failed to create task")
	ErrInternal   = NewCDBError(ER_INTERNAL, "An internal error occered")
	ErrTuneFailed = NewCDBError(ER_TUNE_FAILED, "Run cdbtune failed")
)

type enumTaskStatus struct {
	NotStart        string
	Running         string
	Pause           string
	Stop            string
	NormalFinish    string
	ExceptionFinish string
	Undoed          string
	Crashed         string
	Unknown         string
}
type enumTaskType struct {
	CDBTune    string
	Competitor string
	TCloud     string
}
type enumRwMode struct {
	RW string
	RO string
	WO string
}

var (
	TaskStatus = enumTaskStatus{"not_start", "running", "pause", "stop",
		"normal_finish", "exception_finish", "undoed", "crashed", "unknown"}
	TaskType = enumTaskType{"cdbtune", "competitor", "tcloud"}
	RwMode   = enumRwMode{"rw", "ro", "wo"}
)

// This NullString for mysql row scan
type NullString struct {
	String string
	Valid  bool // Valid is true if Time is not NULL
}

// Scan implements the Scanner interface.
// The value type must be time.Time or string / []byte (formatted time-string),
// otherwise Scan fails.
func (nt *NullString) Scan(value interface{}) (err error) {
	if value == nil {
		nt.String, nt.Valid = "", false
		return
	}
	switch value.(type) {
	case []uint8:
		b := make([]byte, len(value.([]uint8)))
		for i, v := range value.([]uint8) {
			b[i] = byte(v)
		}
		nt.String, nt.Valid = string(b), true
		return
	case string:
		nt.String, nt.Valid = value.(string), true
		return
	}
	nt.Valid = false
	return fmt.Errorf("Can't convert %T to string", value)
}

// Value implements the driver Valuer interface.
func (nt NullString) Value() (driver.Value, error) {
	if !nt.Valid {
		return nil, nil
	}
	return nt.String, nil
}

func SimpleExecScript(bin string, args ...string) (err error, stdout_o string, stderr_o string) {
	var out bytes.Buffer
	var outerr bytes.Buffer
	cmd := exec.Command(bin, args...)
	cmd.Stdout = &out
	cmd.Stderr = &outerr
	TLog.Infof("ExecScript bin: %s, args: %v", bin, args)
	err = cmd.Run()
	TLog.Infof("ExecScript out: %s, err: %s", strings.TrimSpace(out.String()), strings.TrimSpace(outerr.String()))
	return err, strings.TrimSpace(out.String()), strings.TrimSpace(outerr.String())
}
