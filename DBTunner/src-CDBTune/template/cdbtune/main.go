package main

import (
	. "git.code.oa.com/gocdb/base/frame"
	"os"
	"path"
)

func main() {
        //assign config
	config_path := "../etc/" + path.Base(os.Args[0]) + ".conf"

	app := NewApp()
	
	appFrame := NewAppFrame(app, config_path)
	
	if err := appFrame.Init(); err != nil {
		os.Exit(3)
	}
	defer appFrame.Uninit()
	appFrame.Run()
	
}
