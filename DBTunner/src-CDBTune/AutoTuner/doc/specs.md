## Specs

> **Update**: 2018.5.1

### Parameters Setting

* gamma: discount, = 0.99
* batch_size: 128
* update_target: 10 target网络更新频率
* DDPG的tau: 0.001
* Actor Learning Rate: 0.005
* Critic Learning Rate: 0.001



### Implementation Detailes

##### Reward 函数设计

* Version 0.1 简化版

```` python
tps = 并发提高百分比
latency = 延迟下降百分比

reward = 0  # tps - latency
if tps < 0:
    reward += 2 * tps
else:
    reward += tps
if latency > 0:
    reward -= 2 * latency
else:
    reward -= latency
````

* Version 0.2

```` python

def _calculate_reward(delta0, deltat):

    if delta0 > 0:
        _reward = (1+delta0)**2 * (1+deltat)
    else:
        _reward = - (1-delta0)**2 * (1-deltat)
    return _reward

# tps
delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0]))/self.default_externam_metrics[0]
delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0]))/self.last_external_metrics[0]

tps_reward = self._calculate_reward(delta_0_tps, delta_t_tps)

# latency 
delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[1]
delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]

lat_reward = self._calculate_reward(delta_0_lat, delta_t_lat)

reward = tps_reward * 0.4 + 0.6 * lat_reward

````

##### Actor 网络

```` python
Actor(
  (layers): Sequential(
    (0): BatchNorm1d(63, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (1): Linear(in_features=63, out_features=128, bias=True)
    (2): LeakyReLU(negative_slope=0.2)
    (3): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (4): Linear(in_features=128, out_features=128, bias=True)
    (5): Tanh()
    (6): Dropout(p=0.3)
    (7): Linear(in_features=128, out_features=64, bias=True)
    (8): Tanh()
    (9): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (10): Linear(in_features=64, out_features=16, bias=True)
  )
  (act): Sigmoid()
)
````

##### Critic 网络

```` python
Critic(
  (state_input): Linear(in_features=63, out_features=128, bias=True)
  (action_input): Linear(in_features=16, out_features=128, bias=True)
  (act): Tanh()
  (state_bn): BatchNorm1d(63, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layers): Sequential(
    (0): Linear(in_features=256, out_features=256, bias=True)
    (1): LeakyReLU(negative_slope=0.2)
    (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Linear(in_features=256, out_features=64, bias=True)
    (4): Tanh()
    (5): Dropout(p=0.3)
    (6): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Linear(in_features=64, out_features=1, bias=True)
  )
)

````