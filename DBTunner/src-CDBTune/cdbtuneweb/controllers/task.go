package controllers

import (
	"cdbtuneweb/base"
	"fmt"
	"strings"

	. "git.code.oa.com/gocdb/base/public"

	"github.com/astaxie/beego"
)

const remote_add = "http://yt.db.cdbtune.oa.com"

type TaskController struct {
	beego.Controller
}

func (t *TaskController) Post() {
	fmt.Println("task:::", t.Ctx.Request.URL)
	reqURL := strings.TrimPrefix(fmt.Sprint(t.Ctx.Request.URL), "")
	switch reqURL {
	case "/task/query_task_result":
		t.HandleQueryTaskResult()
	case "/task/create_task":
		// app.HandleQueryTask(w, r)
	case "/update_task":
		// app.HandleUpdateTask(w, r)
	case "/query_task_result":
		// app.HandleQueryTaskResult(w, r)
	case "/insert_task_result":
		// app.HandleInsertTaskResult(w, r)
	}

}

func (t *TaskController) HandleQueryTaskResult() {
	bytes, err := base.HttpRedirectPost(remote_add+"/query_task_result", t.Ctx.Request)
	if err == nil {
		t.SetData(string(*bytes))
		t.ServeJSON()
	}
	TLog.Errorf("HandleQueryTaskResult err=%+v", err)
}
