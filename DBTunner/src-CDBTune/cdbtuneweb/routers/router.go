package routers

import (
	"cdbtuneweb/controllers"

	"github.com/astaxie/beego"
)

func init() {
	beego.SetStaticPath("/static*", "static")
	beego.ErrorController(&controllers.ErrorController{})
	beego.Router("/task/*", &controllers.TaskController{})

	// beego.Router("/edit_param.html", &controllers.EditParamController{})
	beego.Router("/*.html", &controllers.MainController{})
	beego.Router("/", &controllers.DefaultController{})

	// beego.Router("/cdbtune", &controllers.ZanjieController{})
}
