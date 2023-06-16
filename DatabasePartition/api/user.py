# import logging
# import random
# from flask import Blueprint, request
# from datetime import datetime, timedelta
# from api.utils.code import ResponseCode
# from api.utils.response import ResMsg
# from api.utils.util import route, PhoneTool
# from api.utils.auth import Auth, login_required
# from api.services.wx_login_or_register import get_access_code, get_wx_user_info, wx_login_or_register
# from api.services.phone_login_or_register import SendSms, phone_login_or_register
# from api.utils.cache import cache
#
# bp = Blueprint("user", __name__, url_prefix='/')
#
# logger = logging.getLogger(__name__)
#
#
# @route(bp, '/login', methods=["POST"])
# def login():
#     """
#     登陆成功获取到数据获取token和刷新token
#     :return:
#     """
#     res = ResMsg()
#     obj = request.get_json(force=True)
#     user_name = obj.get("name")
#     # 未获取到参数或参数不存在
#     if not obj or not user_name:
#         res.update(code=ResponseCode.InvalidParameter)
#         return res.data
#
#     # 生成数据获取token和刷新token
#     access_token, refresh_token = Auth.encode_auth_token(user_id=user_name)
#
#     data = {"access_token": access_token.decode("utf-8"),
#             "refresh_token": refresh_token.decode("utf-8")
#             }
#     res.update(data=data)
#     return res.data
#
#
# @route(bp, '/refreshToken', methods=["GET"])
# @login_required
# def refresh_token():
#     """
#     刷新token，获取新的数据获取token
#     :return:
#     """
#     res = ResMsg()
#     refresh_token = request.args.get("refresh_token")
#     if not refresh_token:
#         res.update(code=ResponseCode.InvalidParameter)
#         return res.data
#     payload = Auth.decode_auth_token(refresh_token)
#     # token被串改或过期
#     if not payload:
#         res.update(code=ResponseCode.PleaseSignIn)
#         return res.data
#
#     # 判断token正确性
#     if "user_id" not in payload:
#         res.update(code=ResponseCode.PleaseSignIn)
#         return res.data
#     # 获取新的token
#     access_token = Auth.generate_access_token(user_id=payload["user_id"])
#     data = {"access_token": access_token.decode(
#         "utf-8"), "refresh_token": refresh_token}
#     res.update(data=data)
#     return res.data
#
#
# @route(bp, '/wxLoginOrRegister', methods=["GET"])
# def wx_login_or_register():
#     """
#     微信登陆注册
#     :return:
#     """
#     res = ResMsg()
#     code = request.args.get("code")
#     flag = request.args.get("flag")
#     # 参数错误
#     if code is None or flag is None:
#         res.update(code=ResponseCode.InvalidParameter)
#         return res.data
#     # 获取微信用户授权码
#     access_code = get_access_code(code=code, flag=flag)
#     if access_code is None:
#         res.update(code=ResponseCode.WeChatAuthorizationFailure)
#         return res.data
#     # 获取微信用户信息
#     wx_user_info = get_wx_user_info(access_data=access_code)
#     if wx_user_info is None:
#         res.update(code=ResponseCode.WeChatAuthorizationFailure)
#         return res.data
#
#     # 验证微信用户信息本平台是否有，
#     data = wx_login_or_register(wx_user_info=wx_user_info)
#     if data is None:
#         res.update(code=ResponseCode.Fail)
#         return res.data
#     res.update(data=data)
#     return res.data
#
#
# @route(bp, '/getVerificationCode', methods=["GET"])
# def test_get_verification_code():
#     """
#     获取手机验证码
#     :return:
#     """
#     now = datetime.now()
#     res = ResMsg()
#
#     category = request.args.get("category", None)
#     # category 参数如下：
#     # authentication: 身份验证
#     # login_confirmation: 登陆验证
#     # login_exception: 登陆异常
#     # user_registration: 用户注册
#     # change_password: 修改密码
#     # information_change: 信息修改
#
#     phone = request.args.get('phone', None)
#
#     # 验证手机号码正确性
#     re_phone = PhoneTool.check_phone(phone)
#     if phone is None or re_phone is None:
#         res.update(code=ResponseCode.MobileNumberError)
#         return res.data
#     if category is None:
#         res.update(code=ResponseCode.InvalidParameter)
#         return res.data
#
#     try:
#         # 获取手机验证码设置时间
#         flag = cache.get(re_phone + 'get_code_expire_time')
#         if flag is not None:
#             flag = datetime.strptime(flag, '%Y-%m-%d %H:%M:%S')
#             # 判断是否重复操作
#             if (flag - now).total_seconds() < 60:
#                 res.update(code=ResponseCode.FrequentOperation)
#                 return res.data
#
#         # 获取随机验证码
#         code = "".join([str(random.randint(0, 9)) for _ in range(6)])
#         template_param = {"code": code}
#         # 发送验证码
#         sms = SendSms(
#             phone=re_phone,
#             category=category,
#             template_param=template_param)
#         sms.send_sms()
#         # 将验证码存入redis，方便接下来的验证
#         cache.set(re_phone, code, timeout=180)
#         # 设置重复操作屏障
#         cache.set(re_phone + "get_code_expire_time", (now + timedelta(minutes=1)).strftime('%Y-%m-%d %H:%M:%S'),
#                   timeout=60)
#         return res.data
#     except Exception as e:
#         logger.exception(e)
#         res.update(code=ResponseCode.Fail)
#         return res.data
#
#
# @route(bp, '/mobileLoginOrRegister', methods=["POST"])
# def mobile_login_or_register():
#     """
#     用户验证码登录或注册
#     :return:
#     """
#     res = ResMsg()
#
#     obj = request.get_json(force=True)
#     phone = obj.get('account', None)
#     code = obj.get('code', None)
#     if phone is None or code is None:
#         res.update(code=ResponseCode.InvalidParameter)
#         return res.data
#     # 验证手机号和验证码是否正确
#     flag = PhoneTool.check_phone_code(phone, code)
#     if not flag:
#         res.update(code=ResponseCode.InvalidOrExpired)
#         return res.data
#
#     # 登陆或注册
#     data = phone_login_or_register(phone)
#
#     if data is None:
#         res.update(code=ResponseCode.Fail)
#         return res.data
#     res.update(data=data)
#     return res.data
