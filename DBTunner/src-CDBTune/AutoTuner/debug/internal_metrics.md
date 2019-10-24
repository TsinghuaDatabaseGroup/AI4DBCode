## Debug - Internal Metrics

**测试内部metric正确性**

> Update: 2018-05-03

```` python

# Default Settings - Raw Data [Metric tps: 1935.956 lat: 50.385 qps: 30976.368]

[  0.00000000e+00   2.34431100e+06   2.31014400e+06   2.91174400e+06
   0.00000000e+00   1.41000000e+02   8.60000000e+01   1.07520000e+07
   1.36533333e+04   6.56250000e+02   8.33333333e-01   1.04791075e+06
   1.00000000e+00   1.04856800e+06   0.00000000e+00   0.00000000e+00
   5.59758000e+06   1.41000000e+02   1.71798692e+10   0.00000000e+00
   4.13000000e+02   0.00000000e+00   0.00000000e+00   1.38707500e+06
   0.00000000e+00   3.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   8.60000000e+01
   8.60000000e+01   1.63840000e+04   4.80000000e+01   1.45300000e+03
   5.00000000e+01   4.50000000e+01   1.35000000e+03   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   9.00000000e+00   1.26000000e+02   0.00000000e+00   3.46000000e+02
   1.41000000e+02   3.46000000e+02   6.91200000e+04   1.74000000e+02
   0.00000000e+00   0.00000000e+00   5.32000000e+00]
  
# Default Settings - Normalized Data

[[-2.02857934  8.93524537 -1.15534545  1.38542205  0.         -1.15535724
   1.39577235 -1.94383543  0.52926338 -1.94383547  0.52923913 -0.25253154
  -1.99806828 -0.27460664  0.          0.          0.43476702 -1.15534245
  -0.27461833  0.          3.23429325  0.          0.         -1.87277185
   0.         -0.89919701  0.          0.          0.          0.          0.
   0.          0.          0.         -0.07691577  1.3898377   1.3898377
   0.31673905 -0.09812058 -1.36927923 -1.83845338  0.1367444  -1.87068569
  -1.75052232  0.          0.          0.          0.          0.          0.
   0.          0.          3.87280799  1.37686913  0.          1.25661902
  -1.15547509  1.25377516  1.48196121  1.12408946 -0.31106618  0.
  -1.89994775]]
  

# Next State - Raw Data [Metric tps:1920.361 lat:51.584 qps:30729.624]

[  0.00000000e+00   2.32873200e+06   2.32652800e+06   2.64448000e+06
   0.00000000e+00   1.42000000e+02   7.80000000e+01   1.07506347e+07
   1.50186667e+04   6.56166667e+02   9.16666667e-01   2.05515883e+06
   1.00000000e+00   2.05581600e+06   0.00000000e+00   0.00000000e+00
   5.55626000e+06   1.42000000e+02   3.36844554e+10   0.00000000e+00
   4.10000000e+02   0.00000000e+00   0.00000000e+00   1.37348200e+06
   0.00000000e+00   3.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.80000000e+01
   7.80000000e+01   1.63840000e+04   5.30000000e+01   1.59000000e+03
   5.30000000e+01   4.60000000e+01   1.38000000e+03   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   8.00000000e+00   1.17000000e+02   0.00000000e+00   3.21000000e+02
   1.42000000e+02   3.21000000e+02   6.40000000e+04   1.65000000e+02
   0.00000000e+00   0.00000000e+00   5.32000000e+02]


# Next State - Normalized Data

[[-2.02857934  8.8623343  -1.15528997  1.0739507   0.         -1.15530176
   1.07920206 -1.94384631  0.78150552 -1.94384635  0.78146972  1.26937005
  -1.99806828  1.23760307  0.          0.          0.41649821 -1.15528697
   1.23774336  0.          3.19598497  0.          0.         -1.87440614
   0.         -0.89919701  0.          0.          0.          0.          0.
   0.          0.          0.         -0.07691577  1.07381234  1.07381234
   0.31673905  0.09001949 -1.35081068 -1.83584867  0.17830805 -1.86979322
  -1.75052232  0.          0.          0.          0.          0.          0.
   0.          0.          3.22275135  1.13821389  0.          1.02042811
  -1.15541962  1.01778898  1.22639363  0.9619734  -0.31106618  0.
  -1.89994775]]

# State - Raw Data [Metric tps:1945.939 lat:50.676 qps:31133.793]

[  0.00000000e+00   2.36629400e+06   2.34291200e+06   2.51289600e+06
   0.00000000e+00   1.43000000e+02   7.40000000e+01   1.07492693e+07
   1.22880000e+04   6.56083333e+02   7.50000000e-01   5.41730917e+05
   1.00000000e+00   5.42388000e+05   0.00000000e+00   0.00000000e+00
   5.64886400e+06   1.43000000e+02   8.88668160e+09   0.00000000e+00
   4.17000000e+02   0.00000000e+00   0.00000000e+00   1.39795600e+06
   0.00000000e+00   5.10000000e+01   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   7.40000000e+01
   7.40000000e+01   1.63840000e+04   4.80000000e+01   1.44000000e+03
   4.80000000e+01   3.70000000e+01   1.11200000e+03   1.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
   8.00000000e+00   1.16000000e+02   0.00000000e+00   3.12000000e+02
   1.43000000e+02   3.12000000e+02   6.34880000e+04   1.64000000e+02
   0.00000000e+00   0.00000000e+00   5.32000000e+02]

# State - Normalized Data

[[-2.02857934  9.03812772 -1.1552345   0.92060178  0.         -1.15524629
   0.92091692 -1.94385718  0.27702123 -1.94385722  0.27700854 -1.01734401
  -1.99806828 -1.0345489   0.          0.          0.45744121 -1.1552315
  -1.03454645  0.          3.28537096  0.          0.         -1.87146363
   0.          0.63746269  0.          0.          0.          0.          0.
   0.          0.          0.         -0.07691577  0.91579966  0.91579966
   0.31673905 -0.09812058 -1.37103172 -1.84018984 -0.19576478 -1.877766
  -1.74422667  0.          0.          0.          0.          0.          0.
   0.          0.          3.22275135  1.11169665  0.          0.93539938
  -1.15536415  0.93283395  1.20083687  0.9439605  -0.31106618  0.
  -1.89994775]]

````


## Internal Metric 信息

* x: 在不同knob和负载下会改变
* xx: 改变比较大
* x (write): 在含有写的workload下会改变


|            Name                  |     Type     |     Value    |
|----------------------------------|--------------|--------------|
| adaptive_hash_searches           |status_counter|x (write)|
| adaptive_hash_searches_btree     |status_counter|x|
| buffer_data_reads                |status_counter|x|
| buffer_data_written              |status_counter|x|
| buffer_pages_created             |status_counter|x (write)|
| buffer_pages_read                |status_counter|x|
| buffer_pages_written             |status_counter|x|
| buffer_pool_bytes_data           |     value    |x|
| buffer_pool_bytes_dirty          |     value    |x|
| buffer_pool_pages_data           |     value    |x|
| buffer_pool_pages_dirty          |     value    |x|
| buffer_pool_pages_free           |     value    |x|
| buffer_pool_pages_misc           |     value    |x|
| buffer_pool_pages_total          |     value    |x|
| buffer_pool_reads                |status_counter|0.00000000e+00|
| buffer_pool_read_ahead           |status_counter|0.00000000e+00|
| buffer_pool_read_ahead_evicted   |status_counter|x|
| buffer_pool_read_requests        |status_counter|x|
| buffer_pool_size                 |     value    |x|
| buffer_pool_wait_free            |status_counter|0.00000000e+00|
| buffer_pool_write_requests       |status_counter|x|
| dml_deletes                      |status_counter|x (write)|
| dml_inserts                      |status_counter|x (write)|
| dml_reads                        |status_counter|x|
| dml_updates                      |status_counter|x (write)|
| file_num_open_files              |     value    |x|
| ibuf_merges                      |status_counter|0.00000000e+00|
| ibuf_merges_delete               |status_counter|0.00000000e+00|
| ibuf_merges_delete_mark          |status_counter|0.00000000e+00|
| ibuf_merges_discard_delete       |status_counter|0.00000000e+00|
| ibuf_merges_discard_delete_mark  |status_counter|0.00000000e+00|
| ibuf_merges_discard_insert       |status_counter|0.00000000e+00|
| ibuf_merges_insert               |status_counter|0.00000000e+00|
| ibuf_size                        |status_counter|x (write)|
| innodb_activity_count            |status_counter|x (write)|
| innodb_dblwr_pages_written       |status_counter|xx|
| innodb_dblwr_writes              |status_counter|xx|
| innodb_page_size                 |status_counter|x|
| innodb_rwlock_s_os_waits         |status_counter|x|
| innodb_rwlock_s_spin_rounds      |status_counter|x|
| innodb_rwlock_s_spin_waits       |status_counter|x|
| innodb_rwlock_x_os_waits         |status_counter|xx|
| innodb_rwlock_x_spin_rounds      |status_counter|x|
| innodb_rwlock_x_spin_waits       |status_counter|x (write)|
| lock_deadlocks                   |    counter   |0.00000000e+00|
| lock_row_lock_current_waits      |status_counter|0.00000000e+00|
| lock_row_lock_time               |status_counter|x (write)|
| lock_row_lock_time_avg           |     value    |0.00000000e+00|
| lock_row_lock_time_max           |     value    |0.00000000e+00|
| lock_row_lock_waits              |status_counter|x (write)|
| lock_timeouts                    |    counter   |0.00000000e+00|
| log_waits                        |status_counter|x|
| log_writes                       |status_counter|x|
| log_write_requests               |status_counter|xx|
| metadata_mem_pool_size           |     value    |x|
| os_data_fsyncs                   |status_counter|xx|
| os_data_reads                    |status_counter|x|
| os_data_writes                   |status_counter|x|
| os_log_bytes_written             |status_counter|xx|
| os_log_fsyncs                    |status_counter|x|
| os_log_pending_fsyncs            |status_counter|x (write)|
| os_log_pending_writes            |status_counter|x (write)|
| trx_rseg_history_len             |     value    |x (write)|


