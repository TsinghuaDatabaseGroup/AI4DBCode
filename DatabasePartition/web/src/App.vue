<template>
  <div class="container">
    <div class="c-flex-row c-justify-content-between c-align-items-center"
         style="margin: 10px 20px; font-weight: bolder; font-size: 30px; color: var(--text_color)">
      <span>DatabasePartition</span>
      <div style="cursor: pointer;" @click="onThemeClick">
        <img v-if="theme === 'light'" class="theme-icon" src="@/assets/night.png" style="width: 32px; height: 32px">
        <img v-else class="theme-icon" src="@/assets/light.png" style="width: 32px; height: 32px">
      </div>
    </div>

    <div class="c-flex-row c-justify-content-between" style="margin: 10px 20px;">
      <div class="c-flex-column c-justify-content-between" style="width: 30%">
        <el-card shadow="always" style="width: 100%;">
          <el-select class="dbSelect" v-model="dbSelect" placeholder="请选择">
            <el-option
                v-for="item in dbOptions"
                :key="item.value"
                :label="item.label"
                :value="item.value">
            </el-option>
          </el-select>
          <el-button type="success" style="margin-left: 20px">LOAD</el-button>
        </el-card>
        <el-card shadow="always" style="width: 100%; margin-top: 20px">
          <div class="c-flex-column c-align-items-center" style="width: 100%;">
            <VChart style="width: 100%; height: 300px;" :option="pieOption" autoresize/>
            <span style="color: var(--text_color); font-size: 16px">
                sql10000条，访问100个表
            </span>
          </div>
        </el-card>
      </div>
      <el-card shadow="always" style="width: calc(70% - 30px); ">
        <div style="overflow-y: scroll; height: 100%; max-height: 45vh; min-height: 400px">
          <div v-for="(item, index) in suggests" class="suggest-item" :key="index">
            <el-checkbox :checked="checkedList.indexOf(index) > -1" @change="onCheckBoxChange(index)"/>
            <div class="c-flex-column" style="margin-left: 10px">
              <span style="margin-bottom: 5px; font-size: 14px">待处理的表: item1,分片键: i id</span>
              <span style="font-size: 14px">分割语句: ALTER TABLE item1 set distributed by(i id);</span>
            </div>
          </div>
        </div>
        <el-button type="success" style="margin: 10px 0 10px 0; float: right">ACTION</el-button>
      </el-card>
    </div>

    <el-card shadow="always" style="margin: 10px 20px;">
      <div class="c-flex-row c-align-items-center" style="width: 100%">

        <div class="c-flex-column c-align-items-center" style="width: 40%">
          <VChart style="width: 100%; height: 300px;" :option="barOption" autoresize/>
          <span style="color: var(--text_color); font-size: 16px">
                Performance
        </span>
        </div>

        <div class="c-flex-column c-align-items-center" style="width: 60%;">
          <div class="c-flex-row c-justify-content-around c-align-items-center" style="width: 100%">
            <div class="c-flex-column c-relative">
              <img class="c-relative" src="@/assets/db.png" style="width: 200px; height: 200px">
              <span class="c-absolute"
                    style="top: 5px; left: 0; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                1.9TB
              </span>
              <span class="c-absolute"
                    style="top: 5px; left: 120px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                1TB
              </span>
              <span class="c-absolute"
                    style="top: 80px; left: 60px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                0.1TB
              </span>
            </div>
            <img src="@/assets/arrow.png" style="width: 20%; height: 30px">
            <div class="c-flex-column c-relative">
              <img class="c-relative" src="@/assets/db.png" style="width: 200px; height: 200px">
              <span class="c-absolute"
                    style="top: 5px; left: 0; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                1TB
              </span>
              <span class="c-absolute"
                    style="top: 5px; left: 120px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                1TB
              </span>
              <span class="c-absolute"
                    style="top: 80px; left: 60px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;">
                1TB
              </span>
            </div>
          </div>
          <span style="color: var(--text_color); font-size: 16px">
                Data Distribution
        </span>
        </div>
      </div>


    </el-card>

  </div>
</template>

<script>

import {use} from 'echarts/core'
import {CanvasRenderer} from 'echarts/renderers'
import {BarChart, PieChart} from 'echarts/charts'
import 'vue-organization-chart/dist/orgchart.css'
import {GridComponent, LegendComponent, TitleComponent, TooltipComponent} from 'echarts/components'
import {default as VChart, THEME_KEY} from 'vue-echarts'

use([
  CanvasRenderer,
  PieChart,
  BarChart,
  TooltipComponent,
  TitleComponent,
  GridComponent,
  LegendComponent
])

export default {
  name: 'Index',
  components: {
    VChart
  },
  provide: {
    [THEME_KEY]: 'light'
  },
  data() {
    return {
      dbOptions: [
        {label: 'Postgres', value: 1},
        {label: 'Mysql', value: 2}
      ],
      dbSelect: 1,
      schema: "",
      wow: null,
      innerVisible: false,
      costVisible: false,
      theme: '',
      pieOption: {
        title: {
          show: false,
          left: 'center'
        },
        tooltip: {
          trigger: 'item'
        },
        series: [
          {
            data: [{name: "a", value: 0.02}, {name: "b", value: 0.1}, {name: "c", value: 0.02}, {name: "d", value: 0.1}],
            type: 'pie',
            center: ['50%', '50%'],
            itemStyle: {
              borderWidth: 0,
              borderColor: '#ffffff'
            }
          }
        ]
      },
      barOption: {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          }
        },
        legend: {
          show: true,
          data: ['a', 'b'],
          textStyle: {
            color: "auto"
          }
        },
        grid: {
          left: '0',
          top: '40px',
          right: '0',
          bottom: '0',
          containLabel: true
        },
        yAxis: [
          {
            type: 'value'
          }
        ],
        xAxis: {
          type: 'category',
          boundaryGap: true,
          axisLabel: {
            interval: 0
          },
          axisTick: {
            alignWithLabel: true
          }
        },
        series: [
          {
            itemStyle: {
              normal: {
                color: function (params) {
                  var colorList = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff3366', '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff3366']
                  return colorList[params.dataIndex]
                }
              }
            },
            name: 'a',
            type: 'bar',
            colorBy: 'data',
            barWidth: 'auto',
            data: [{name: "a", value: 0.02}, {name: "b", value: 0.1}, {name: "c", value: 0.02}, {name: "d", value: 0.1}]
          },
          {
            itemStyle: {
              normal: {
                color: function (params) {
                  var colorList = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff3366', '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#ff3366']
                  return colorList[params.dataIndex]
                }
              }
            },
            name: 'b',
            type: 'bar',
            colorBy: 'data',
            barWidth: 'auto',
            data: [{name: "a", value: 0.02}, {name: "b", value: 0.1}, {name: "c", value: 0.02}, {name: "d", value: 0.1}]
          }
        ]
      },
      checkedList: [],
      suggests: ["", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", ""]
    }
  },
  mounted() {
    var theme = sessionStorage.getItem('theme-key') || 'light'
    this.theme = theme
    document.body.setAttribute("theme-mode", theme);
  },
  destroyed() {
    if (this.explainSqlTypedObj) {
      this.explainSqlTypedObj.destroy()
    }
    if (this.explainNodeTypedObj) {
      this.explainNodeTypedObj.destroy()
    }
  },
  methods: {
    onThemeClick() {
      var theme = sessionStorage.getItem('theme-key') === 'dark' ? 'light' : 'dark'
      this.theme = theme
      sessionStorage.setItem('theme-key', theme)
      document.body.setAttribute("theme-mode", theme);
    },
    onCheckBoxChange(index) {
      if (this.checkedList.indexOf(index) > -1) {
        this.checkedList.slice(this.checkedList.indexOf(index), 1)
      } else {
        this.checkedList.push(index)
      }
    },
    handleCopy(value) {
      const that = this
      this.$copyText(value).then(function () {
        that.$message.success(' Copied to clipboard！')
      }, function () {
        that.$message.error('Copy failed, please press Ctrl+C to copy！')
      })
    }
  }
}
</script>

<style>

:root {
  --body_background_color: white;
  --textarea_inner_background: white;
  --textarea_inner_color: #333333;
  --dialog_background_color: #ffffff;
  --dialog_color: #303133;
  --select_background_color: #f5f5f5;
  --sql_container_background_color: #f5f5f5;
  --sql_container_color: #000000;
  --text_color: #333333;
  --text_grey: #666666;
  --card_background_color: #ffffff;
  --theme_icon_contnet: "\eaf8";
  --card_shadow_color: 0 2px 12px 0 rgb(0 0 0 / 10%);;
}

:root body[theme-mode="dark"] {
  --body_background_color: #1a2234;
  --textarea_inner_background: #222f3e;
  --textarea_inner_color: #FFFFFF;
  --dialog_background_color: #1a2134;
  --dialog_color: #FFFFFF;
  --select_background_color: #222f3e;
  --sql_container_background_color: #222f3e;
  --sql_container_color: #ffffff;
  --text_color: #ffffff;
  --text_grey: #ffffff;
  --card_background_color: #1d273c;
  --theme_icon_contnet: "\eb7d";
  --card_shadow_color: 0 2px 12px 0 rgb(155 155 155 / 10%);
}

body {
  background-color: var(--body_background_color) !important;
  transition: color 0.5s, background-color 0.5s;
}

.orgchart-container {
  width: 100% !important;
  height: calc(70vh - 160px) !important;
  margin: auto;
}

.orgchart > table:first-child {
  margin: 40px auto 0 auto;
}

.el-form-item__label {
  color: var(--text_color);
}

.el-upload__tip {
  color: var(--text_grey);
}

.input .el-textarea__inner {
  height: 100%;
  background: var(--textarea_inner_background) !important;
  color: var(--textarea_inner_color) !important;
  transition: color 0.5s, background-color 0.5s;
  font-size: 20px;
}

.el-textarea.is-disabled .el-textarea__inner {
  background-color: #666666 !important;
  color: lawngreen !important;
}

.report-container .el-dialog__body {
  padding: 0 10px 10px 10px !important;
}

.el-dialog__header {
  background-color: var(--body_background_color) !important;
  transition: color 0.5s, background-color 0.5s;
}

.el-dialog__body {
  background-color: var(--dialog_background_color) !important;
  transition: color 0.5s, background-color 0.5s;
}

.el-dialog__title {
  color: var(--dialog_color);
}

.report-container .el-dialog {
  margin-bottom: 0 !important;
}

.el-loading-mask {
  background-color: var(--dialog_background_color);
}

.el-tabs__header .el-tabs__item.is-active {
  background-color: var(--body_background_color);
  border: none;
}

.el-tabs__item {
  background-color: var(--dialog_background_color);
  color: var(--text_color);
  border: none;
}

.el-tabs__nav-wrap::after {
  display: none;
}

.el-tabs--border-card > .el-tabs__header {
  background-color: var(--dialog_background_color);
  border: none;
}

.el-dialog__close {
  color: var(--dialog_color);
  transition: color 0.5s, background-color 0.5s;
}

.el-card {
  background: var(--dialog_background_color);
  border-color: var(--dialog_background_color);
  transition: color 0.5s, background-color 0.5s, border-color 0.5s;
}

.el-card.is-always-shadow {
  box-shadow: var(--card_shadow_color);
}

.el-select {
  background-color: var(--card_background_color);
  width: calc(100% - 90px);
  height: 38px;
}

.dbSelect .el-input__inner {
  background-color: var(--select_background_color);
  border: none;
  color: #409EFF;
  font-size: 20px;
  padding-left: 5px;
  height: 38px;
}

</style>

<style scoped>

/*最外层透明*/
::v-deep .el-table,
::v-deep .el-table__expanded-cell {
  background-color: var(--sql_container_background_color) !important;
}

/* 表格内背景颜色 */
::v-deep .el-table th,
::v-deep .el-table tr,
::v-deep .el-table td {
  background-color: var(--sql_container_background_color) !important;
  color: var(--text_color);
}

/*去除底边框*/
::v-deep.el-table td.el-table__cell {
  border: 1px solid var(--text_color);
}

::v-deep.el-table th.el-table__cell.is-leaf {
  border: 1px solid var(--text_color);
}

.container {
  display: flex;
  flex-direction: column;
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
}

.tab-container {
  display: flex;
  flex-direction: row;
  justify-content: left;
}

.tab-item {
  display: flex;
  flex-direction: row;
  text-align: center;
  color: var(--text_color);
  padding: 0 10px;
  cursor: pointer;
}

.tab-item-active {
  color: #409EFF;
}

.suggest-item {
  color: var(--text_color);
  background: var(--sql_container_background_color);
  width: calc(100% - 20px);
  position: relative;
  text-align: left;
  margin: 10px 0;
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 10px;
  font-size: 16px;
}

.sql-container {
  width: calc(100% - 20px);
  background: var(--sql_container_background_color);
  border-radius: 4px;
  white-space: pre;
  padding: 10px;
  color: var(--sql_container_color);
  font-size: 1em;
  position: relative;
  text-align: left;
  height: calc(100% - 20px);
  overflow-y: auto;
  margin: 10px 0;
  transition: color 0.5s, background-color 0.5s;
}

.copy-ontainer {
  position: absolute;
  right: 0;
  top: 6px;
  cursor: pointer;
  color: #409EFF;
}

.el-icon-document-copy:hover {
  color: #409EFF;
}

.el-icon-document-copy:active {
  color: #000000;
}

.schemaTreeChart {
  width: 100%;
  height: 100%;
}

.customSchemaTreeChart {
  width: 100%;
  height: calc(100% - 80px);
}

</style>


