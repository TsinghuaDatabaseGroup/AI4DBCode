package main

import (
	_ "cdbtuneweb/routers"

	"github.com/astaxie/beego"
	// "github.com/astaxie/beego"
)

func main() {
	// knobs := base.ReadKnobs("knob_metric.txt")
	// fmt.Println(knobs)
	beego.Run()
}
