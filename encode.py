import ui
# Example Usage
api_keys = {
  "type": "service_account",
  "project_id": "carl-9b3f3",
  "private_key_id": "9b99c0622a751f146e86f4fc9bd467f03c46e464",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC/xl8fs+wSkuhn\nol+hA7Eh93oliLOY1tKpEcAvJ8LeoKgOOCAj6SBzUjLAtvoT3UmtCTJWPuOOymox\nyn9i15xz1Us7cpKnGOnABeqWxVgNiN9qtJ3s8zaZw0bJ4gux0t6rSG+ZdEfa8Nqr\na4OYHGG6GgDGCGyzi2rcdWkpFrkHQ4prhpCiTKLlA2zVUu0Z8B1Vkia4n4sv7GmE\nroGOuORRrPdKqM9OhHNxsS98/K14cjma7T/YQMUvvrwfn2cRw4xNJk34yiqKaAW3\nESvezllln9Oj6gsxJNxj0FGveLFIyrZKEInliLwEcvtf7P91EVB6ifPY8Zscebjp\nYWXBpBifAgMBAAECggEAA99SOxD7qqxYu677AK9TEP0u9Dm8a/ulKVDNxqX7jgSZ\nyspgq8Ogqh4gyY9oS/pjQHaNE6dqxOQFckKagTkaqaQcRB5eI7hMZVo5OxjPsbDv\n10xqW/HVSmmNBl/lczkaJ0SWgaISyV2lAg0bt83q6JfEfZCHUWDL+Zj3/00Rfg9a\nwuGA+uhQl7y6z4ZnLba7ls6OqQ19YB3+D3Sn5wgCVop40uotggjNHP83pTR0raez\n0dsMP0GinnS0Xa75by/b89wWDs22XT1VoY7kiQ1chH9LVRLodHF/JpOzYfUzNi/6\ng0oQU7aq9mYSGXabsg0BT94csLyRord00CpdTXbjkQKBgQDxklfLoZVW+7Tof28p\ng5moPA+d2fuHphaVEOZJQxOTKVAc0yA7hl3Oy95YKtcZwVLuoCiZwfu/yT/Bqdy/\nHijmov6KeATfnIS4Vu9RFeJ3pMsCD2XAwwnNiaZqKoyD+SbqS/y68xGgA/4en7LI\nQ55DkZZtwGVSmVvMLWCSRFutvQKBgQDLOqFyxWYDzv/esTi/DQE3SNAEn+79bI2S\n6ioZ63QLaLCMC2TmiavF9dkqS0q9/sEekkQMbZpr6WZQywGiPu29Ue8puxu7l2i+\nb6LNxRKGus5nPWrYrXAgP6AL0zjQs78An7DV9FYXih9jvaPfoDnIhHfdgB996fNT\n2jYxSBx/iwKBgGM4foMO/S3a+LU+EkR03xnwgWGOdPeESmYzqMKSoGmjYFpWhTit\nub4EknCYN/1GIHAOrF5rBKIDYQKEaDy/gIEqlW3+WdIWkZS9cFJXsMr/jrpr5JlG\nArK/RPD6RPi3zzoQt995ktWsjiW55k7HJywNkkHF8lf40XGNecrZ9OXpAoGADvvj\nq3KicuVwOBsY8/0hedIEhFLGbCj0x0A8mmyhwbWWTr0IU3cTEyVtPZEPkbKWyoo+\nOixallo/EPXmyO+a17qSx6DkCpC/SEsy1bkSBJ0BWttMZW1kNvx58GVCayDVlFYx\n05SQRGwKpG/3BSXrHL2nmM05hS8aobQVCs0mTMECgYEAluZ4T/K6ebLPmGx14Wgh\nKKL8LBDdjKlGZVBw/irLtkKk/Rl6d7MGlK74F+8WFzRoh3xtALb0aBOmtI/meFMb\nDdvuuagDEamJRJ0kzsw9wnQdRXXslxv7IWFQ9K1rZDR7HPFXb3sSLRqsIkMICDcl\nDsQ7vD/Sa8VoKIB6P4XN8YE=\n-----END PRIVATE KEY-----\n",
  "client_email": "firebase-adminsdk-9ta75@carl-9b3f3.iam.gserviceaccount.com",
  "client_id": "111105703320827232295",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": f'https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-9ta75%40carl-9b3f3.iam.gserviceaccount.com',
  "universe_domain": "googleapis.com"
}
other_api_keys = {"openai_api_key" : "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",
                  "genai_api_key" : "AIzaSyCi06GaEKyCKqjctTG-bldhQfcmmFBlXKA"}
ui.embed_api_keys('./assets/logo_carl.png', other_api_keys, './assets/logo_carl_new.png')
extracted_keys = ui.extract_api_keys('./assets/logo_carl.png_new')
print(extracted_keys)