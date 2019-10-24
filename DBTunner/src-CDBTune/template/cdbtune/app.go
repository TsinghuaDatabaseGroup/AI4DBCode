package main 

import (
	. 	"git.code.oa.com/gocdb/base/public"
	tconf "git.code.oa.com/gocdb/base/config"
)

type cdbtune struct {
	ch  chan interface{}
}

func NewApp() *cdbtune {
        //new the program app
        return &cdbtune{}
}

func (a *cdbtune) GetVersion() string {
	return "0.1.0"
}

func (a *cdbtune) LoadUserConf(conf tconf.Configer, reload bool) error {
	if !reload {

	}
	return nil
}
	
func (a *cdbtune) OnStartApp() error {
	TLog.Info("cdbtune is start")
	return nil
}

func (a *cdbtune) OnStopApp() {
	TLog.Info("cdbtune is stop")
}


func (a *cdbtune) GetEventChan() <-chan interface{} {
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
