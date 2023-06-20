<template>
  <div class="container">
    <div
        class="c-flex-row c-justify-content-between c-align-items-center"
        style="margin: 10px 20px; font-weight: bolder; font-size: 30px; color: var(--text_color); overflow: auto"
    >
    </div>

    <div class="c-flex-row c-justify-content-between" style="margin: 10px 20px;">
      <div class="c-flex-column" style="width: 30%; min-width: 300px; margin-right: 10px">
        <el-card shadow="always" style="width: 100%; ">
          <el-select v-model="dbSelect" class="dbSelect" placeholder="请选择">
            <el-option
                v-for="item in datasetOptions"
                :key="item.value"
                :label="item.name"
                :value="item.value"
            />
          </el-select>
          <el-button type="success" style="margin-left: 20px" @click="onLoadClcik()">LOAD</el-button>
        </el-card>
        <el-card class="c-relative" shadow="always" style="width: 100%; margin-top: 10px;" v-loading="loading1">
          <div class="c-relative c-flex-column" style="width: 100%;">
            <span style="color: var(--text_color); font-size: 16px">Data Statistics</span>
            <VChart style="width: 100%; height: 340px;" :option="pieOption" autoresize />
          </div>
          <div class="c-relative c-flex-column" style="width: 100%;">
            <span style="color: var(--text_color); font-size: 16px">Query Statistics</span>
            <VChart style="width: 100%; height: 330px;" :option="column2GraphOption" autoresize />
          </div>
          <el-button class="c-relative" type="success" style="margin-bottom: 10px; float: right" @click="onPartitonClcik()">PARTITION
          </el-button>
          <div v-if="hideStep1" class="Mask" />
        </el-card>
      </div>

      <div class="c-relative c-flex-column" style="width: calc(70% - 30px);">
        <el-card class="c-relative" shadow="always" style="width: 100%;" v-loading="loading2">
          <span style="color: var(--text_color); font-size: 16px">Selected Keys</span>
          <div style="overflow-y: scroll; height: 380px">
            <div v-for="(item, index) in recommends" :key="index" class="suggest-item">
              <el-checkbox :checked="checkedList.indexOf(index) > -1" @change="onCheckBoxChange(index)" />
              <div class="c-flex-column" style="margin-left: 10px; font-size: 20px; color: var(--text_grey)">
                <span>Table: <span style="color: var(--text_color)">{{ item.table }}</span></span>
                <span>Partition Key: <span style="color: var(--text_color)">{{ item.column }}</span></span>
                <span style="font-size: 20px; color: var(--text_grey)">Statement: <span style="color: var(--text_color)">{{ item.partition_sql }}</span></span>
              </div>
            </div>
          </div>
          <el-button class="c-relative" :disabled="!canAction" type="success" style="margin: 10px 0 10px 0; float: right" @click="onActionClick()">ACTION
          </el-button>
          <div v-if="hideStep2" class="Mask" />
        </el-card>


        <el-card class="c-relative" shadow="always" style="margin-top: 10px; height: 380px" v-loading="loading3">
          <span style="color: var(--text_color); font-size: 16px">Estimated Partition Results</span>
          <div class="c-relative c-flex-row c-align-items-center" style="width: 100%; height: 380px">
            <div class="c-flex-column c-align-items-center" style="width: 100%;">
              <div class="c-flex-row c-justify-content-around c-align-items-center" style="width: 100%;">
                <div class="c-flex-column c-relative c-align-items-center c-justify-content-center">
                  <img class="c-relative" src="@/assets/db.png" style="width: 200px; height: 200px">
                  <span
                      class="c-absolute"
                      style="top: 5px; left: 0; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    {{ beforeNode1Points }}
                  </span>
                  <span
                      class="c-absolute"
                      style="top: 5px; left: 120px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    {{ beforeNode2Points }}
                  </span>
                  <span
                      class="c-absolute"
                      style="top: 80px; left: 60px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    Master
                  </span>
                  <span style="margin-top: 10px; color: var(--text_color)">Latency: {{ beforePerformance }}</span>
                </div>
                <div class="c-flex-column c-relative c-align-items-center c-justify-content-center" style="width: 20%;">
                  <img src="@/assets/arrow.png" style="width: 100%; height: 30px">
                  <span style="color: var(--text_color); margin-top: 10px">After Partitioning</span>
                </div>
                <div class="c-flex-column c-relative c-align-items-center c-justify-content-center">
                  <img class="c-relative" src="@/assets/db.png" style="width: 200px; height: 200px">
                  <span
                      class="c-absolute"
                      style="top: 5px; left: 0; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    {{ afterNode1Points }}
                  </span>
                  <span
                      class="c-absolute"
                      style="top: 5px; left: 120px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    {{ afterNode2Points }}
                  </span>
                  <span
                      class="c-absolute"
                      style="top: 80px; left: 60px; color: #FFFFFF; font-size: 18px; width: 80px; text-align: center;"
                  >
                    Master
                  </span>
                  <span style="margin-top: 10px; color: var(--text_color)">Latency: {{ afterPerformance }}</span>
                </div>
              </div>
            </div>
          </div>
          <div v-if="hideStep3" class="Mask" />
        </el-card>
      </div>
    </div>

  </div>
</template>

<script>

import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { PieChart, GraphChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TitleComponent, TooltipComponent } from 'echarts/components'
import { default as VChart, THEME_KEY } from 'vue-echarts'
import { dataset, distribution, preview, recommend } from '@/api/api'

use([
  CanvasRenderer,
  PieChart,
  GraphChart,
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
      datasetOptions: [],
      dbSelect: undefined,
      schema: '',
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
            data: [
              {
                'columns': 9,
                'value': '203',
                'name': 'orders'
              },
              {
                'columns': 16,
                'value': '872',
                'name': 'lineitem'
              },
              {
                'columns': 8,
                'value': '28',
                'name': 'customer'
              },
              {
                'columns': 7,
                'value': '0.208',
                'name': 'supplier'
              }
            ],
            label: {
              alignTo: 'edge',
              minMargin: 5,
              edgeDistance: 10,
              lineHeight: 15
            },
            radius: ['0', '60%'],
            type: 'pie',
            center: ['50%', '50%'],
            itemStyle: {
              borderWidth: 0,
              borderColor: '#ffffff'
            }
          }
        ]
      },
      column2GraphOption: {
        tooltip: {},
        series: [
          {
            name: '',
            type: 'graph',
            layout: 'circular',
            data: [{
              'category': 0,
              'id': 'l_orderkey',
              'name': 'l_orderkey',
              'symbolSize': 60,
              'value': 0.2499493807554245
            },
              {
                'category': 1,
                'id': 'l_quantity',
                'name': 'l_quantity',
                'symbolSize': 20,
                'value': 8.331646313308738e-06
              },
              {
                'category': 2,
                'id': 'l_shipdate',
                'name': 'l_shipdate',
                'symbolSize': 20,
                'value': 0.0004209147591609508
              },
              {
                'category': 3,
                'id': 'o_orderkey',
                'name': 'o_orderkey',
                'symbolSize': 60,
                'value': 1.0
              },
              {
                'category': 4,
                'id': 'o_custkey',
                'name': 'o_custkey',
                'symbolSize': 60,
                'value': 0.06666400283575058
              },
              {
                'category': 5,
                'id': 'o_orderdate',
                'name': 'o_orderdate',
                'symbolSize': 20,
                'value': 0.0016039999900385737
              },
              {
                'category': 6,
                'id': 'c_custkey',
                'name': 'c_custkey',
                'symbolSize': 60,
                'value': 1.0
              }],
            links: [
              {
                'lineStyle': {
                  'width': 6.04
                },
                'source': 'l_orderkey',
                'target': 'o_orderkey',
                'value': 604.0
              },
              {
                'lineStyle': {
                  'width': 20
                },
                'source': 'l_shipdate',
                'target': 'o_orderdate',
                'value': 365792.0
              },
              {
                'lineStyle': {
                  'width': 1.89
                },
                'source': 'o_custkey',
                'target': 'c_custkey',
                'value': 189.0
              }],
            categories: [
              0,
              1,
              2,
              3,
              4,
              5,
              6
            ],
            center: ['50%', '50%'],
            roam: true,
            force: {
              repulsion: 100
            },
            lineStyle: {
              color: 'source',
              curveness: 0.5
            },
            label: {
              show: true,
              color: 'auto',
            },
            itemStyle: {},
            emphasis: {
              focus: 'adjacency',
              lineStyle: {
                width: 10
              }
            }
          }
        ],
        animationDuration: 1500,
        animationEasingUpdate: 'quinticInOut'
      },
      checkedList: [],
      distributions: [],
      recommends: [
        {
          'column': 'l_orderkey,l_shipdate',
          'partition_sql': 'ALTER TABLE lineitem set distributed by(l_orderkey,l_shipdate);',
          'table': 'lineitem'
        },
        {
          'column': 'o_orderkey,o_custkey,o_orderdate',
          'partition_sql': 'ALTER TABLE orders set distributed by(o_orderkey,o_custkey,o_orderdate);',
          'table': 'orders'
        },
        {
          'column': 'c_custkey',
          'partition_sql': 'ALTER TABLE customer set distributed by(c_custkey);',
          'table': 'customer'
        }
      ],
      beforeNode1Points: 32542,
      beforeNode2Points: 32539,
      afterNode1Points: 32553,
      afterNode2Points: 32528,
      beforePerformance: 2.5126,
      afterPerformance: 2.5118,
      hideStep1: true,
      hideStep2: true,
      hideStep3: true,
      loading1: false,
      loading2: false,
      loading3: false
    }
  },
  computed: {
    canAction() {
      return this.checkedList.length > 0
    }
  },
  mounted() {
    var theme = sessionStorage.getItem('theme-key') || 'light'
    this.theme = theme
    document.body.setAttribute('theme-mode', theme)
    this.column2GraphOption.series[0].label.color = theme === 'dark' ? '#fff' : '#000'
    this.getDataset()

    this.pieOption.tooltip.formatter = function(param, ticket, callback) {
      const result =
          `${param.marker}Tabel:&nbsp${param.data['name']}</br>&nbsp&nbsp&nbspColumns:&nbsp${param.data['columns']}</br>&nbsp&nbsp&nbspSize:&nbsp${(param.data['value'])}MB</br>`
      setTimeout(function() {
        // 仅为了模拟异步回调
        callback(ticket, result)
      }, 100)
      return 'loading...'
    }

    this.pieOption.series[0].label.formatter = function(data) {
      return data.name + '\n' + data.data.columns + ' columns'
    }
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
      document.body.setAttribute('theme-mode', theme)
      this.column2GraphOption.series[0].label.color = theme === 'dark' ? '#fff' : '#000'
    },
    onCheckBoxChange(index) {
      if (this.checkedList.indexOf(index) > -1) {
        this.checkedList.splice(this.checkedList.indexOf(index), 1)
      } else {
        this.checkedList.push(index)
      }
    },
    getDataset() {
      dataset().then(res => {
        this.datasetOptions = res.data || []
        if (this.datasetOptions.length > 0) {
          this.dbSelect = this.datasetOptions[0].value
        }
      })
    },
    onLoadClcik() {
      if (!this.dbSelect) {
        this.$message.warning('Please select a dataset！')
        return
      }
      this.getDistributions()
    },
    onPartitonClcik() {
      this.getRecommend()
    },
    getDistributions() {
      this.loading1 = true
      distribution({ dataset: this.dbSelect }).then(res => {
        const distributions = res.data.distributions || []
        this.distributions = distributions.map(item => {
          if (item.size.slice(-2).toLowerCase() === 'kb') {
            item.size = parseFloat(item.size.toLowerCase().replace(' kb', '')) / 1024
          } else if (item.size.slice(-2).toLowerCase() === 'mb') {
            item.size = parseFloat(item.size.toLowerCase().replace(' mb', ''))
          } else if (item.size.slice(-2).toLowerCase() === 'gb') {
            item.size = parseFloat(item.size.toLowerCase().replace(' mb', '')) * 1024
          }
          return {
            name: item.table_name,
            value: item.size,
            columns: item.columns
          }
        })
        this.pieOption.series[0].data = this.distributions

        const column2Graph = res.data.column2Graph || {}

        this.column2GraphOption.series[0].data = column2Graph.nodes
        this.column2GraphOption.series[0].links = column2Graph.links
        this.column2GraphOption.series[0].categories = column2Graph.categories
        this.hideStep1 = false
      }).finally(() => {
        this.loading1 = false
      })
    },
    getRecommend() {
      this.loading2 = true
      recommend({ dataset: this.dbSelect }).then(res => {
        this.recommends = res.data || []
        this.hideStep2 = false
      }).finally(() => {
        this.loading2 = false
      })
    },
    onActionClick() {
      var recommend = this.recommends.filter((item, index) => {
        return this.checkedList.indexOf(index) > -1
      }) || []
      var partitionKeys = {}
      recommend.forEach(item => {
        partitionKeys[item.table] = item.column
      })
      this.loading3 = true
      preview({ partition_keys: partitionKeys }).then(res => {
        this.hideStep3 = false
        this.beforeNode1Points = res.data.distribution.before.node_0.points
        this.beforeNode2Points = res.data.distribution.before.node_1.points
        this.afterNode1Points = res.data.distribution.after.node_0.points
        this.afterNode2Points = res.data.distribution.after.node_1.points
        this.beforePerformance = res.data.performance.before
        this.afterPerformance = res.data.performance.after
      }).finally(() => {
        this.loading3 = false
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

.Mask {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  backdrop-filter: saturate(180%) blur(10px);
  z-index: 101;
  transition: all 0.3s linear;
  -moz-transition: all 0.3s linear;
  -webkit-transition: all 0.3s linear;
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
  font-size: 16px;
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

