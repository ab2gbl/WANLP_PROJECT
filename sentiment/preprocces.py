
import re
from pyarabic.araby import strip_tashkeel, tokenize,normalize_hamza

SHADDA = u'\u0651'

Ashkal = (u'\u064b', u'\u064c', u'\u064d', u'\u064e', u'\u064f', u'\u0650', u'\u0652', u'\u0651')
def shakl(text):
    if not text:
        return text
    else:
        for char in Ashkal:
            text = text.replace(char, '')
    return text


HAMZA = u'\u0621'
ALEF_MAD = u'\u0622'
ALEF = u'\u0627'

def alif(word):
        
    HAMZAT = (u'\u0621', u'\u0624', u'\u0626', u'\u0654', u'\u0655', u'\u0625', u'\u0623',)
    
    HAMZAT_PATTERN = re.compile(u"[" + u"".join(HAMZAT) + u"]", re.UNICODE)
    
    if word.startswith(ALEF_MAD):
        if len(word) >= 3 and (word[1] not in Ashkal) and (word[2] == SHADDA or len(word) == 3):
            word = HAMZA + ALEF + word[1:]
        else:
            word = HAMZA + HAMZA + word[1:]
    word = word.replace(ALEF_MAD, HAMZA + HAMZA)
    word = HAMZAT_PATTERN.sub(HAMZA, word)
    return word
