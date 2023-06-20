import axios from 'axios'
import {Message} from 'element-ui'
import baseUrl from "@/api/baseUrl";

const service = axios.create({
  baseURL: baseUrl
})

axios.defaults.timeout = 1200000
axios.defaults.timeoutErrorMessage = 'axios  timeout'

// request interceptor
service.interceptors.request.use(
  config => {
    // do something before request is sent
    if (config.method === 'get') {
      config.params = {
        ...config.params
      }
    } else {
      if (config.headers['Content-Type'] !== 'multipart/form-data') {
        config.data = {
          ...config.data,
        }
      }
    }
    return config
  },
  error => {
    // do something with request error
    console.log(error) // for debug
    return Promise.reject(error)
  }
)


service.interceptors.response.use(
  response => {
    const res = response.data
    console.log("res:", res)
    if (res.code !== 0) {
      Message({
        message: res.msg || 'Error',
        type: 'error',
        duration: 5 * 1000
      })
      return Promise.reject(new Error(res.msg || 'Error'))
    } else {
      return res
    }
  },
  error => {
    Message({
      message: error.msg,
      type: 'error',
      duration: 5 * 1000
    })
    return Promise.reject(error)
  }
)

export function post (url, data) {
  return service({
    url: url,
    method: 'post',
    data: data
  })
}

export default service
