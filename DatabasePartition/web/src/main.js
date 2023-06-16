import Vue from 'vue'
import App from './App.vue'
import '@/assets/common.css'
import {Button, Card, Input, Message, Loading, Dialog, Tooltip, Table, TableColumn, Steps, Step, Select, Option, Checkbox} from 'element-ui';

Vue.prototype.$ELEMENT = { size: 'small', zIndex: 3000 };
Vue.use(Input);
Vue.use(Button);
Vue.use(Loading);
Vue.use(Dialog);
Vue.use(Card);
Vue.use(Tooltip);
Vue.use(Table);
Vue.use(TableColumn);
Vue.use(Steps);
Vue.use(Step);
Vue.use(Select);
Vue.use(Option);
Vue.use(Checkbox);

Vue.component(Message.name, Message);
Vue.prototype.$message = Message;

import VueClipboard from 'vue-clipboard2'
VueClipboard.config.autoSetContainer = true // add this line
Vue.use(VueClipboard)

Vue.config.productionTip = false;

new Vue({
  render: h => h(App),
}).$mount('#app');
