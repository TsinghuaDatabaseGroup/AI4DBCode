const schema = [{"table":"customer1","rows":0,"columns":[{"name":"c_id","index":true,"type":"integer"},{"name":"c_d_id","type":"smallint"},{"name":"c_w_id","type":"smallint"},{"name":"c_first","type":"character varying"},{"name":"c_middle","type":"character"},{"name":"c_last","type":"character varying"},{"name":"c_street_1","type":"character varying"},{"name":"c_street_2","type":"character varying"},{"name":"c_city","type":"character varying"},{"name":"c_state","type":"character"},{"name":"c_zip","type":"character"},{"name":"c_phone","type":"character"},{"name":"c_since","type":"character"},{"name":"c_credit","type":"character"},{"name":"c_credit_lim","type":"bigint"},{"name":"c_discount","type":"numeric"},{"name":"c_balance","type":"numeric"},{"name":"c_ytd_payment","type":"numeric"},{"name":"c_payment_cnt","type":"smallint"},{"name":"c_delivery_cnt","type":"smallint"},{"name":"c_data","type":"character"}]},{"table":"district1","rows":0,"columns":[{"name":"d_id","index":true,"type":"smallint"},{"name":"d_w_id","type":"smallint"},{"name":"d_name","type":"character varying"},{"name":"d_street_1","type":"character varying"},{"name":"d_street_2","type":"character varying"},{"name":"d_city","type":"character varying"},{"name":"d_state","type":"character"},{"name":"d_zip","type":"character"},{"name":"d_tax","type":"numeric"},{"name":"d_ytd","type":"numeric"},{"name":"d_next_o_id","type":"integer"}]},{"table":"history1","rows":0,"columns":[{"name":"h_c_id","index":true,"type":"integer"},{"name":"h_c_d_id","type":"smallint"},{"name":"h_c_w_id","type":"smallint"},{"name":"h_d_id","type":"smallint"},{"name":"h_w_id","type":"smallint"},{"name":"h_date","type":"character"},{"name":"h_amount","type":"numeric"},{"name":"h_data","type":"character varying"}]},{"table":"item1","rows":0,"columns":[{"name":"i_id","index":true,"type":"integer"},{"name":"i_im_id","type":"integer"},{"name":"i_name","type":"character varying"},{"name":"i_price","type":"numeric"},{"name":"i_data","type":"character varying"}]},{"table":"new_orders1","rows":0,"columns":[{"name":"no_o_id","type":"integer"},{"name":"no_d_id","type":"smallint"},{"name":"no_w_id","type":"smallint"}]},{"table":"order_line1","rows":0,"columns":[{"name":"ol_o_id","type":"integer"},{"name":"ol_d_id","type":"smallint"},{"name":"ol_w_id","type":"smallint"},{"name":"ol_number","type":"smallint"},{"name":"ol_i_id","type":"integer"},{"name":"ol_supply_w_id","type":"smallint"},{"name":"ol_delivery_d","type":"character"},{"name":"ol_quantity","type":"smallint"},{"name":"ol_amount","type":"numeric"},{"name":"ol_dist_info","type":"character"}]},{"table":"orders1","rows":0,"columns":[{"name":"o_id","index":true,"type":"integer"},{"name":"o_d_id","type":"smallint"},{"name":"o_w_id","type":"smallint"},{"name":"o_c_id","type":"integer"},{"name":"o_entry_d","type":"character"},{"name":"o_carrier_id","type":"smallint"},{"name":"o_ol_cnt","type":"smallint"},{"name":"o_all_local","type":"smallint"}]},{"table":"stock1","rows":0,"columns":[{"name":"s_i_id","index":true,"type":"integer"},{"name":"s_w_id","type":"smallint"},{"name":"s_quantity","type":"smallint"},{"name":"s_dist_01","type":"character"},{"name":"s_dist_02","type":"character"},{"name":"s_dist_03","type":"character"},{"name":"s_dist_04","type":"character"},{"name":"s_dist_05","type":"character"},{"name":"s_dist_06","type":"character"},{"name":"s_dist_07","type":"character"},{"name":"s_dist_08","type":"character"},{"name":"s_dist_09","type":"character"},{"name":"s_dist_10","type":"character"},{"name":"s_ytd","type":"numeric"},{"name":"s_order_cnt","type":"smallint"},{"name":"s_remote_cnt","type":"smallint"},{"name":"s_data","type":"character varying"}]},{"table":"warehouse1","rows":0,"columns":[{"name":"w_id","type":"smallint"},{"name":"w_name","type":"character varying"},{"name":"w_street_1","type":"character varying"},{"name":"w_street_2","type":"character varying"},{"name":"w_city","type":"character varying"},{"name":"w_state","type":"character"},{"name":"w_zip","type":"character"},{"name":"w_tax","type":"numeric"},{"name":"w_ytd","type":"numeric"}]}]

const schemaTreeChartOption = {
  "tooltip": {
    "trigger": "item",
      "triggerOn": "mousemove",
      "formatter": "{b0}<br /> {c0}"
  },
  "series": [
    {
      "type": "tree",
      "id": 0,
      "name": "tree",
      "data": [{
        "name": "schema",
        "children": [{
          "name": "customer",
          "value": "rows:1000",
          "children": [{"name": "c_custkey", "value": "type:integer"}, {"name": "c_name", "value": "type:character varying"}, {
            "name": "c_address",
            "value": "type:character varying"
          }, {"name": "c_nationkey", "value": "type:integer"}, {"name": "c_phone", "value": "type:character"}, {
            "name": "c_acctbal",
            "value": "type:numeric"
          }, {"name": "c_mktsegment", "value": "type:character"}, {"name": "c_comment", "value": "type:character varying"}, {
            "name": "c_null",
            "value": "type:character varying"
          }],
          "collapsed": true
        }, {
          "name": "lineitem",
          "value": "rows:1000",
          "children": [{"name": "l_orderkey", "value": "type:integer"}, {"name": "l_partkey", "value": "type:integer"}, {
            "name": "l_suppkey",
            "value": "type:integer"
          }, {"name": "l_linenumber", "value": "type:integer"}, {"name": "l_quantity", "value": "type:numeric"}, {
            "name": "l_extendedprice",
            "value": "type:numeric"
          }, {"name": "l_discount", "value": "type:numeric"}, {"name": "l_tax", "value": "type:numeric"}, {
            "name": "l_returnflag",
            "value": "type:character"
          }, {"name": "l_linestatus", "value": "type:character"}, {"name": "l_shipdate", "value": "type:date"}, {
            "name": "l_commitdate",
            "value": "type:date"
          }, {"name": "l_receiptdate", "value": "type:date"}, {"name": "l_shipinstruct", "value": "type:character"}, {
            "name": "l_shipmode",
            "value": "type:character"
          }, {"name": "l_comment", "value": "type:character varying"}, {"name": "l_null", "value": "type:character varying"}],
          "collapsed": true
        }, {
          "name": "nation",
          "value": "rows:1000",
          "children": [{"name": "n_nationkey", "value": "type:integer"}, {"name": "n_name", "value": "type:character"}, {
            "name": "n_regionkey",
            "value": "type:integer"
          }, {"name": "n_comment", "value": "type:character varying"}, {"name": "n_null", "value": "type:character varying"}],
          "collapsed": true
        }, {
          "name": "ol_rule_list",
          "value": "rows:1000",
          "children": [{"name": "ol_id", "value": "type:integer"}, {"name": "rownum", "value": "type:integer"}],
          "collapsed": true
        }, {
          "name": "orders",
          "value": "rows:1000",
          "children": [{"name": "o_orderkey", "value": "type:integer"}, {"name": "o_custkey", "value": "type:integer"}, {
            "name": "o_orderstatus",
            "value": "type:character"
          }, {"name": "o_totalprice", "value": "type:numeric"}, {"name": "o_orderdate", "value": "type:date"}, {
            "name": "o_orderpriority",
            "value": "type:character"
          }, {"name": "o_clerk", "value": "type:character"}, {"name": "o_shippriority", "value": "type:integer"}, {
            "name": "o_comment",
            "value": "type:character varying"
          }, {"name": "o_null", "value": "type:character varying"}],
          "collapsed": true
        }, {
          "name": "part",
          "value": "rows:1000",
          "children": [{"name": "p_partkey", "value": "type:integer"}, {"name": "p_name", "value": "type:character varying"}, {
            "name": "p_mfgr",
            "value": "type:character"
          }, {"name": "p_brand", "value": "type:character"}, {"name": "p_type", "value": "type:character varying"}, {
            "name": "p_size",
            "value": "type:integer"
          }, {"name": "p_container", "value": "type:character"}, {"name": "p_retailprice", "value": "type:numeric"}, {
            "name": "p_comment",
            "value": "type:character varying"
          }, {"name": "p_null", "value": "type:character varying"}],
          "collapsed": true
        }, {
          "name": "partsupp",
          "value": "rows:1000",
          "children": [{"name": "ps_partkey", "value": "type:integer"}, {"name": "ps_suppkey", "value": "type:integer"}, {
            "name": "ps_availqty",
            "value": "type:integer"
          }, {"name": "ps_supplycost", "value": "type:numeric"}, {"name": "ps_comment", "value": "type:character varying"}, {
            "name": "ps_null",
            "value": "type:character varying"
          }],
          "collapsed": true
        }, {
          "name": "region",
          "value": "rows:1000",
          "children": [{"name": "r_regionkey", "value": "type:integer"}, {"name": "r_name", "value": "type:character"}, {
            "name": "r_comment",
            "value": "type:character varying"
          }, {"name": "r_null", "value": "type:character varying"}],
          "collapsed": true
        }, {
          "name": "supplier",
          "value": "rows:1000",
          "children": [{"name": "s_suppkey", "value": "type:integer"}, {"name": "s_name", "value": "type:character"}, {
            "name": "s_address",
            "value": "type:character varying"
          }, {"name": "s_nationkey", "value": "type:integer"}, {"name": "s_phone", "value": "type:character"}, {
            "name": "s_acctbal",
            "value": "type:numeric"
          }, {"name": "s_comment", "value": "type:character varying"}, {"name": "s_null", "value": "type:character varying"}],
          "collapsed": true
        }]
      }],
      "top": "1%",
      "left": "70px",
      "bottom": "1%",
      "right": "100px",
      "symbolSize": 7,
      "lineStyle": {
        "width": 2
      },
      "label": {
        "position": "left",
        "align": "right",
        "fontSize": 12
      },
      "leaves": {
        "label": {
          "position": "right",
          "verticalAlign": "middle",
          "align": "left"
        }
      },
      "emphasis": {
        "focus": "descendant"
      },
      "expandAndCollapse": true,
      "animationDuration": 550,
      "animationDurationUpdate": 750
    }
  ]
}

const customSchemaTreeChartOption = {
  tooltip: {
    trigger: 'item',
      triggerOn: 'mousemove',
      formatter: '{b0}<br /> {c0}'
  },
  series: [
    {
      type: 'tree',
      id: 0,
      name: 'tree',
      data: [],
      top: '1%',
      left: '70px',
      bottom: '1%',
      right: '100px',
      symbolSize: 7,
      lineStyle: {
        width: 2
      },
      label: {
        position: 'left',
        verticalAlign: 'middle',
        align: 'right',
        fontSize: 12
      },
      leaves: {
        label: {
          position: 'right',
          verticalAlign: 'middle',
          align: 'left'
        }
      },
      emphasis: {
        focus: 'descendant'
      },
      expandAndCollapse: true,
      animationDuration: 550,
      animationDurationUpdate: 750
    }
  ]
}


const workload = [
  "SELECT c_discount, c_last, c_credit, w_tax FROM customer1, warehouse1 WHERE c_w_id = w_id AND c_id < ceil(random()*3000);"]

export {
  schema,
  schemaTreeChartOption,
  customSchemaTreeChartOption,
  workload,
}
