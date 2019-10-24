package base

import (
	"bytes"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"strings"
	"time"

	. "git.code.oa.com/gocdb/base/public"

	"github.com/pquerna/ffjson/ffjson"
)

func HttpPost(url string, event interface{}) (*[]byte, error) {
	var bytes []byte
	var eventStr string
	if eventBytes, err := ffjson.Marshal(event); err != nil {
		TLog.Errorf("format event to json string failed, err=%s", err)
		return &bytes, err
	} else {
		eventStr = string(eventBytes)
	}

	TLog.Tracef("Curl %s, param: %+v", url, event)
	TLog.Debugf("eventStr=%s", eventStr)
	client := &http.Client{}
	request, err := http.NewRequest("POST", url, strings.NewReader(eventStr))
	if err != nil {
		err = ErrHttpReq.AddErrMsg("create new request for %s, param: %s, ERROR: %s", url, eventStr, err)
		return &bytes, err
	}

	//request.Header.Add("Host", "") ooops...
	resp, err := client.Do(request)
	if err != nil {
		err = ErrHttpReq.AddErrMsg("do send request for %s, param: %s, ERROR: %s", url, eventStr, err)
		return &bytes, err
	}

	defer resp.Body.Close()
	bytes, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		TLog.Errorf("Get %s Response, ERROR: %s", url, err)
		err = ErrHttpResp.AddErrMsg("Get %s Response, ERROR: %s", url, err)
		return &bytes, err
	}
	TLog.Infof("response status code=%d content=%s", resp.StatusCode, Bytes2String(bytes))

	return &bytes, nil
}

func HttpRedirectPost(remote_addr string, req *http.Request) (*[]byte, error) {
	client := &http.Client{}
	var rst []byte

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	for k, v := range req.Form {
		writer.WriteField(k, v[0])
	}
	writer.Close()
	client.Timeout = time.Second * 5
	req2, err := http.NewRequest(req.Method, remote_addr, bytes.NewReader(body.Bytes()))
	if err != nil {
		return &rst, ErrHttpReq.AddErrMsg("do create request for %s,  ERROR: %s", remote_addr, err)
	}
	req2.Header.Set("Content-Type", writer.FormDataContentType())

	// request
	resp, err := client.Do(req2)
	if err != nil {
		return &rst, ErrHttpReq.AddErrMsg("do send request for %s, ERROR: %s", remote_addr, err)
	}
	defer resp.Body.Close()
	rst, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		TLog.Errorf("Get %s Response, ERROR: %s", remote_addr, err)
		err = ErrHttpResp.AddErrMsg("Get %s Response, ERROR: %s", remote_addr, err)
		return &rst, err
	}
	return &rst, err
}
