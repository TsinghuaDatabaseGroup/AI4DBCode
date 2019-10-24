package main

import (
	"net/http"
	"strconv"

	. "git.code.oa.com/gocdb/base/public"
)

type ProtCommonRsp struct {
	Errno int         `json:"errno"`
	Error string      `json:"error"`
	Data  interface{} `json:"data"`
}

func SendRsp(w http.ResponseWriter, data interface{}, err error) error {
	var rsp ProtCommonRsp

	if err == nil {
		rsp.Errno = 0
		rsp.Error = ""
		rsp.Data = data
	} else if e, ok := err.(*CDBError); ok {
		rsp.Errno = e.Errno()
		rsp.Error = e.Error()
		rsp.Data = data
	} else {
		rsp.Errno = ER_OUTER
		rsp.Error = err.Error()
		rsp.Data = data
	}
	err2 := SendHttpJsonRsp(w, rsp)
	if err2 != nil {
		TLog.Errorf("SendHttpJsonRsp failed +%v", err2)
	}
	return err2
}

//=============================== http ===============================
func (app *TuneServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	/*
		if !app.online { //服务未启动
			TLog.Error("http request arrived but server is not online!")
			var rsp ProtCommonRsp
			rsp.SendErrRsp(w, ErrServerOffline)
			return
		}
	*/
	TLog.Info(r.URL.Path)
	switch r.URL.Path {
	case "/create_task":
		app.HandleCreateTask(w, r)
	case "/query_task":
		app.HandleQueryTask(w, r)
	case "/update_task":
		app.HandleUpdateTask(w, r)
	case "/query_task_result":
		app.HandleQueryTaskResult(w, r)
	case "/insert_task_result":
		app.HandleInsertTaskResult(w, r)
	default:
		http.NotFound(w, r)
		return
	}
}

func hasErr(w http.ResponseWriter, err error) bool {
	if err != nil {
		err := SendRsp(w, nil, err)
		return err == nil
	}
	return false
}

func isEmpty(r *http.Request, fields ...string) (bool, error) {
	for _, v := range fields {
		if len(r.FormValue(v)) == 0 {
			TLog.Errorf("request [%s] form param empty", v)
			err := ErrHttpReq.AddErrMsg("request [%s] form param empty", v)
			return false, err
		}
	}
	return true, nil
}

func getIntField(r *http.Request, field string) (int64, error) {
	ret, err := strconv.Atoi(r.FormValue(field))
	if err != nil {
		TLog.Errorf("client request [%s] not int type err=%+v", field, err)
		return -1, ErrHttpReq.AddErrMsg("client request [%s] not int type err=%+v", field, err)
	}
	return int64(ret), nil
}

//create a new task
func (app *TuneServer) HandleCreateTask(w http.ResponseWriter, r *http.Request) {

	fields := []string{"name", "creator", "task_type", "rw_mode", "run_mode"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	ti, err := CreateTask(app,
		r.FormValue("name"),
		r.FormValue("creator"),
		r.FormValue("task_type"),
		r.FormValue("rw_mode"),
		r.FormValue("run_mode"),
		16,
	)
	if !hasErr(w, err) {
		SendRsp(w, ti, nil)
	}
	//TODO 基于channel 运行
	go ti.Run(app)
}
func (app *TuneServer) HandleQueryTask(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	// var timeBase time.Time
	if !hasErr(w, err) {
		tInfo := &TaskInfo{}
		err := app.QueryByIndex(TB_TASK, "task_id", tid, tInfo)
		if !hasErr(w, err) {
			SendRsp(w, tInfo, nil)
		}
	}
}
func (app *TuneServer) HandleUpdateTask(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	errMsg := r.FormValue("error")
	if !hasErr(w, err) {
		tInfo := &TaskInfo{}
		err := app.QueryByIndex(TB_TASK, "task_id", tid, tInfo)
		if !hasErr(w, err) {
			if len(errMsg) == 0 {
				err = tInfo.SetFinished(app)
			} else {
				err = tInfo.SetErr(app, ErrTuneFailed.AddErrMsg(errMsg))
			}
			if !hasErr(w, err) {
				SendRsp(w, tInfo, nil)
			}
		}
	}
}
func (app *TuneServer) HandleQueryTaskResult(w http.ResponseWriter, r *http.Request) {
	tid, err := getIntField(r, "task_id")
	if !hasErr(w, err) {
		tr := &TaskResult{}
		rst, err := app.DBQuery("*", TB_TASK_RESULT, "task_id = ?", tr, tid)
		if !hasErr(w, err) {
			SendRsp(w, rst, nil)
		}
	}
}
func (app *TuneServer) HandleInsertTaskResult(w http.ResponseWriter, r *http.Request) {
	fields := []string{"task_id", "knobs_detail", "tps", "qps", "rt", "score"}
	if ok, err := isEmpty(r, fields...); !ok {
		SendRsp(w, nil, err)
		return
	}
	tr := &TaskResult{}
	err := tr.Insert(app,
		r.FormValue("task_id"),
		r.FormValue("knobs_detail"),
		r.FormValue("tps"),
		r.FormValue("qps"),
		r.FormValue("rt"),
		r.FormValue("score"),
	)
	if !hasErr(w, err) {
		SendRsp(w, tr, nil)
	}
}

//end of http service
