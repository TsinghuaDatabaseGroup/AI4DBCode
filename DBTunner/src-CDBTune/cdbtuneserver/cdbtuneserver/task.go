package main

import (
	"fmt"
	"time"

	. "git.code.oa.com/gocdb/base/go-sql-driver/mysql"
)

// type NullTime mysql.NullTime

type TaskInfo struct {
	TaskId     int64      `json:task_id`
	Name       string     `json:name`
	Creator    string     `json:creator`
	TaskType   string     `json:task_type`
	RwMode     string     `json:rw_mode`
	RunMode    string     `json:run_mode`
	Status     string     `json:status`
	Threads    int64      `json:threads`
	Error      NullString `json:error`
	CreateTime NullTime   `json:create_time`
	StartTime  NullTime   `json:start_time`
	EndTime    NullTime   `json:end_time`
}
type TbModels struct {
	ModelId      int64    `json:model_id`
	MysqlVersion string   `json:mysql_version`
	Dimension    int64    `json:dimension`
	Knobs        int64    `json:knobs`
	RwType       string   `json:rw_type`
	Method       string   `json:method`
	Position     string   `json:position`
	CreateTime   NullTime `json:create_time`
}

type TaskResult struct {
	ResultId    int64   `json:result_id`
	TaskId      int64   `json:task_id`
	KnobsDetail string  `json:knobs_detail`
	Tps         float64 `json:tps`
	Qps         float64 `json:qps`
	Rt          float64 `json:rt`
	Score       float64 `json:score`
}

func (t *TaskResult) Insert(app *TuneServer, task_id, knobs_detail, tps, qps, rt, score string) error {
	sql := "insert into tb_task_result(task_id,knobs_detail,tps,qps,rt,score) values(%s,'%s',%s,%s,%s,%s)"
	sql = fmt.Sprintf(sql, task_id, knobs_detail, tps, qps, rt, score)
	rID, err := app.DBInsert(sql)
	if err == nil {
		t.ResultId = rID
	}
	return err
}

func (t *TaskInfo) Insert(app *TuneServer) error {
	sql := "insert into tb_task(name,creator,task_type,rw_mode,run_mode,threads) values(?,?,?,?,?,?)"
	tId, err := app.DBInsert(sql, t.Name, t.Creator, t.TaskType, t.RwMode, "sysbench", t.Threads)
	if err == nil {
		t.TaskId = tId
	}
	return err
}

func (t *TaskInfo) Run(app *TuneServer) error {
	err := app.DBUpdate(TB_TASK, "status = ? , start_time = ?", "task_id = ?", TaskStatus.Running, time.Now(), t.TaskId)
	if err != nil {
		t.SetErr(app, err)
		return err
	}

	cmd := "cd /usr/local/cdbtune/tuner && python evaluate.py --task_id %d --inst_id %d --model_id %d --host %s"

	cmd = fmt.Sprintf(cmd, t.TaskId, 20001, 1001, "10.249.84.215:8080")

	err, _, _ = SimpleExecScript("sh", "-c", cmd)
	// err, s_out, s_err := public.SimpleExecScript("ssh", "root@127.0.0.1", "echo 0")
	if err != nil {
		t.SetErr(app, err)
		return err
	}
	return nil
}

func (t *TaskInfo) SetErr(app *TuneServer, err error) error {
	return app.DBUpdate(TB_TASK, "status = ? , error = ?", "task_id = ?", TaskStatus.Pause, err.Error(), t.TaskId)
}

func (t *TaskInfo) SetFinished(app *TuneServer) error {
	return app.DBUpdate(TB_TASK, "status = ? , end_time = ?", "task_id = ?", TaskStatus.NormalFinish, time.Now(), t.TaskId)
}

func CreateTask(app *TuneServer, name, creator, task_type, rw_mode, run_mode string, threads int64) (*TaskInfo, error) {
	tm := NullTime{}
	t := &TaskInfo{0, name, creator, task_type, rw_mode, run_mode, "not_started", threads, NullString{}, tm, tm, tm}
	return t, t.Insert(app)
}
