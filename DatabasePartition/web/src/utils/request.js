import axios from 'axios'
import {Message} from 'element-ui'
import baseUrl from "@/utils/baseUrl";

const service = axios.create({
  baseURL: baseUrl,
  headers: {
    'Content-Type': 'multipart/form-data'
  }
})

axios.defaults.timeout = 120000
axios.defaults.timeoutErrorMessage = 'axios  timeout'

service.interceptors.response.use(
  response => {
    const res = response.data
    console.log("res:", res)
    if (!res.status) {
      Message({
        message: res.message || 'Error',
        type: 'error',
        duration: 5 * 1000
      })
      return Promise.reject(new Error(res.message || 'Error'))
    } else {
      return res
    }
  },
  error => {
    Message({
      message: error.message,
      type: 'error',
      duration: 5 * 1000
    })
    return Promise.reject(error)
  }
)

export function post(url, data, config = {}) {
  return service({
    url: url,
    method: 'post',
    data: stringify(data),
    timeout: config.timeout,
    config: config
  })
}


function stringify (data) {
  const formData = new FormData()
  for (const key in data) {
    // eslint-disable-next-line no-prototype-builtins
    if (data.hasOwnProperty(key)) {
      if (data[key]) {
        if (data[key].constructor === Array) {
          if (data[key][0]) {
            if (data[key][0].constructor === Object) {
              formData.append(key, JSON.stringify(data[key]))
            } else {
              data[key].forEach((item, index) => {
                formData.append(key + `[${index}]`, item)
              })
            }
          } else {
            formData.append(key + '[]', '')
          }
        } else if (data[key].constructor === Object) {
          formData.append(key, JSON.stringify(data[key]))
        } else {
          formData.append(key, data[key])
        }
      } else {
        if (data[key] === 0) {
          formData.append(key, 0)
        } else {
          formData.append(key, '')
        }
      }
    }
  }
  return formData
}

export function get(url, data, config = {}) {
  return service({
    url: url,
    method: 'get',
    data: data,
    timeout: config.timeout,
    config: config
  })
}
export default service
