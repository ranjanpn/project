from translate import Translator
translator= Translator(from_lang="german",to_lang="kn")
translation = translator.translate("Guten Morgen")
print (translation)