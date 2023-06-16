import { post } from '@/utils/request'

export async function distributionKeyParser(args) {
  return await post('/distribution_key_parser', args, { timeout: 120000 })
}
