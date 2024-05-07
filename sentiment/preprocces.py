
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


'''
def remove_emojis(data) :
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', data)



def remove_stopwords(text):
    # Tokenize the text
    tokens = text.split()
    
    # Get NLTK's Arabic stop words
    arabic_stopwords = set(stopwords.words('arabic'))
    
    # Remove stop words
    filtered_tokens = [word for word in tokens if word.lower() not in arabic_stopwords]
    
    # Join the filtered tokens back into a sentence
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
'''