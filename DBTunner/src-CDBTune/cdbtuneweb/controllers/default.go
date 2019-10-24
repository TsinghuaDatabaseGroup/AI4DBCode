package controllers

import (
	"strings"

	. "git.code.oa.com/gocdb/base/public"

	"github.com/astaxie/beego"
)

type DefaultController struct {
	beego.Controller
}

func (this *DefaultController) Get() {
	this.Ctx.Redirect(302, "/login.html")
}

type MainController struct {
	beego.Controller
}

func (c *MainController) HasParam(fields ...string) bool {
	for _, v := range fields {
		if len(c.GetString(v)) == 0 {
			TLog.Errorf("request [%s] form param empty", v)
			err := ErrHttpReq.AddErrMsg("request [%s] form param empty", v)
			return c.HasErr(err)
		}
	}
	return false
}
func (c *MainController) HasErr(err error) bool {
	if err != nil {
		TLog.Errorf("check found some error %+v", err)
		err := ErrHttpReq.AddErrMsg("check found some error %+v", err)
		c.Data["Content"] = err.Error()
		c.Abort("510")
		return true
	}
	return false
}

func (c *MainController) Get() {
	var err error
	reqURL := strings.TrimPrefix(c.Ctx.Input.URL(), "/")
	c.TplName = reqURL
	rUsers := c.Ctx.Request.Header["Staffname"]
	rUser := "unknow"
	if len(rUsers) > 0 {
		rUser = rUsers[0]
	}
	if len(reqURL) == 0 {
		c.TplName = "login.html"
	}
	c.Data["HomePage"] = c.TplName == "login.html"
	c.Data["url"] = reqURL
	c.Data["user"] = rUser
	switch reqURL {
	case "show_result.html":
		c.Data["TaskId"], err = c.GetInt("task_id")
	}
	c.HasErr(err)
}

type ErrorController struct {
	beego.Controller
}

func (c *ErrorController) Error404() {
	c.TplName = "404.html"
}
func (c *ErrorController) Error510() {
	c.TplName = "510.html"
}
