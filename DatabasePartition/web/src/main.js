import Vue from 'vue'
import App from './App.vue'
import '@/assets/common.css'
import {Button, Card, Input, Message, Loading, Tooltip, Select, Option, Checkbox} from 'element-ui';

Vue.prototype.$ELEMENT = { size: 'small', zIndex: 3000 };
Vue.use(Input);
Vue.use(Button);
Vue.use(Loading);
Vue.use(Card);
Vue.use(Tooltip);
Vue.use(Select);
Vue.use(Option);
Vue.use(Checkbox);

Vue.component(Message.name, Message);
Vue.prototype.$message = Message;

Vue.config.productionTip = false;

new Vue({
  render: h => h(App),
}).$mount('#app');
