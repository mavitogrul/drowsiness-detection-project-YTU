from twilio.rest import Client

account_sid = 'ACa0e7cfbae23320c42cdf035119498461'
auth_token = 'bae25a76c0645db79df3588c377c124c'

client = Client(account_sid, auth_token)

message = client.messages.create(
         body='Sürücünün uykulu olduğu tespit edildi ! Dikkatli olun !',
         from_='+16075369130',
         to='+905314983563')

