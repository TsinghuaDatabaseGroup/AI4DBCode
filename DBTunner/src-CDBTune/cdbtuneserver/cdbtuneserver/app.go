package main

import (
	"database/sql"
	"fmt"
	"time"

	tconf "git.code.oa.com/gocdb/base/config"
	"git.code.oa.com/gocdb/base/go-sql-driver/mysql"
	. "git.code.oa.com/gocdb/base/public"
	_ "git.code.oa.com/gocdb/base/service/ha"
	_ "git.code.oa.com/gocdb/base/service/http"
)

//map from old exception feature id to new alarm type level and
type CompatibleExeceptionMap struct {
	level    string
	alarmkey string
}

type TuneServer struct {
	ch chan interface{}

	//conf object
	conf tconf.Configer

	//mode
	mode string

	//status related
	online bool

	/**config related*/
	//db
	dsn       string  //db配置
	conn      *sql.DB //连接池对象
	max_opens int     //最大连接数
	max_idles int     //最大空闲数
	ping_cnt  int     //初始化时，最多尝试N次ping，不通则退出

	//common field
	// support_plat map[string]AlarmInterface

	//alarm_meta
	// alarm_meta *AlarmMeta

	//TODO to be removed
	alarm_url string
}

func NewApp() *TuneServer {
	//new the program app
	appNew := &TuneServer{}
	return appNew
}

func (app *TuneServer) GetVersion() string {
	return "0.1.0"
}

func (app *TuneServer) OnStartApp() error {
	TLog.Info("TuneServer is start")
	err := app.CreateConnection()

	return err
}

func (app *TuneServer) OnStopApp() {
	TLog.Info("TuneServer is stop")
}

func (app *TuneServer) LoadUserConf(conf tconf.Configer, reload bool) error {
	var err error
	if !reload { //只能在启动时加载的变量
		if err = app.loadDbConf(conf); err != nil {
			TLog.Errorf("error=%+v LoadDbConf error", err)
			return err
		}
	}
	app.conf = conf
	TLog.Infof("LoadUserConf finished succ")

	return nil
}

//below db connection related
func (app *TuneServer) GetEventChan() <-chan interface{} {
	return nil
}

func (app *TuneServer) CreateConnection() error {
	var err error
	if app.conn, err = sql.Open("mysql", app.dsn); err != nil { //创建db
		TLog.Errorf("open dsn %s failed: %s", app.dsn, err)
		return err
	}
	if app.max_opens > 0 {
		app.conn.SetMaxOpenConns(app.max_opens)
	}
	if app.max_idles >= 0 {
		app.conn.SetMaxIdleConns(app.max_idles)
	}
	mysql.SetLogger(TLog) //设置go-mysql-driver的日志

	for i := 0; ; {
		if err = app.conn.Ping(); err == nil {
			TLog.Info("connect db ok!")
			break
		}
		if i++; i >= app.ping_cnt {
			TLog.Errorf("Ping failed over %d time(s)", app.ping_cnt)
			return err
		}
		TLog.Warnf("Ping %d time(s): %s, try again", i, err)
		time.Sleep(time.Second * 3)
	}
	return nil
}

func (app *TuneServer) loadDbConf(conf tconf.Configer) error {
	var err error
	var user, passwd, db, ip, port string

	if user, err = GetConfNonEmptyString(conf, "mysql::user"); err != nil {
		return err
	}
	passwd = conf.String("mysql::passwd")

	if ip, err = GetConfNonEmptyString(conf, "mysql::host"); err != nil {
		return err
	}
	if port, err = GetConfNonEmptyString(conf, "mysql::port"); err != nil {
		return err
	}
	if db, err = GetConfNonEmptyString(conf, "mysql::db"); err != nil {
		return err
	}
	app.dsn = fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true&loc=Local", user, passwd, ip, port, db) //本地时区
	app.max_opens = conf.DefaultInt("mysql::max_open", 0)
	app.max_idles = conf.DefaultInt("mysql::max_idle", -1)
	app.ping_cnt = conf.DefaultInt("mysql::ping_cnt", 3)

	TLog.Infof(">>>DB DSN: %s, open: %d, idle: %d, ping: %d",
		app.dsn, app.max_opens, app.max_idles, app.ping_cnt)
	return nil
}

/*OPTION:  implement the service handler you need
  detail in gocdb/base/frame/app_frame.go

  service.Listener: 需要向appframe发送消息时实现

  HttpService:
    ServeHTTP(w http.ResponseWriter, r *http.Request)

  HaService:
    BeLeader() error
    BeFollower() error
    DoUpgrade() error
    DoDegrade() error

  GrpcService:
    RegisterPb(s *grpc.Server) error

  BiClient:
    GetServer() (server string, retryStart bool)
    RegisterPacketSender(sender PacketSender)
    GetPacketHandler() PacketHandler

  BiServer:
    RegisterPacketSender(sender PacketSender)
    GetPacketHandler() PacketHandler

  ...
*/

/*
 below logic added code
*/

//=============================== ha ===============================
func (app *TuneServer) BeLeader() error {
	TLog.Info("I'am leader!!!")
	return app.StartServe()
}

func (app *TuneServer) BeFollower() error {
	TLog.Info("I'am follower,just wait...")
	return nil
}

func (app *TuneServer) DoUpgrade() error {
	TLog.Info("promoted to leader")
	return app.StartServe()
}

func (app *TuneServer) DoDegrade() error {
	TLog.Info("degrate to follower, have to restart...")
	app.online = false
	return ErrNeedRestart
}

func (app *TuneServer) StartServe() error {
	app.online = true //设置online，正式提供服务
	return nil
}

//end of ha service
