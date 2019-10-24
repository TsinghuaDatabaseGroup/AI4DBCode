package base

import (
	"bufio"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
)

type Knob struct {
	Result KnobResult
	Param  KnobParam
}

type KnobResult struct {
	QPS   float64
	TPS   float64
	Delay float64
}

type KnobParam struct {
	Params map[string]interface{}
}

func NewEmptyKnob() Knob {
	knob := Knob{KnobResult{}, KnobParam{map[string]interface{}{}}}
	return knob
}
func NewKnob(QPS, TPS, Delay float64, param map[string]interface{}) Knob {
	knob := Knob{KnobResult{QPS, TPS, Delay}, KnobParam{param}}
	return knob
}
func ReadKnobs(fileName string) (knobs []Knob) {
	f, err := os.Open(fileName)
	defer f.Close()
	if !HasErr(err) {
		buf := bufio.NewReader(f)
		for {
			line, err := buf.ReadString('\n')
			if err != nil {
				break
			}
			field := strings.Split(line, ",")
			if len(field) == 4 {
				param := map[string]interface{}{}
				paramField := strings.Split(field[3], "#")
				for _, kv := range paramField {
					kvField := strings.Split(kv, ":")
					param[kvField[0]] = kvField[1]
				}
				tps, _ := strconv.ParseFloat(field[0], 64)
				delay, _ := strconv.ParseFloat(field[1], 64)
				qps, _ := strconv.ParseFloat(field[2], 64)
				knobs = append(knobs, NewKnob(tps, delay, qps, param))
			}
		}
	}
	if err == io.EOF {
		err = nil
	}
	if HasErr(err) {
		knobs = []Knob{}
	}
	return
}

//CheckErr log error
func HasErr(err error) bool {
	if err != nil {
		log.Println(err)
		return true
	}
	return false
}
