import { post } from '@/api/request'

export async function dataset(args) {
  return await post('/partition/dataset', args, { timeout: 1200 })
}

export async function distribution(args) {
  return await post('/partition/distribution', args, { timeout: 1200 })
}

export async function recommend(args) {
  return await post('/partition/recommend', args, { timeout: 1200 })
}

export async function preview(args) {
  return await post('/partition/preview', args, { timeout: 1200 })
}
