from tokenization_qwen import QWenTokenizer
import tiktoken

tokenizer = QWenTokenizer('cl100k_base.tiktoken')
# enc = tiktoken.get_encoding("cl100k_base")


def tokenize_test(text):
    print('input text:', text)
    print('tokens:', tokenizer.tokenize(text))
    print('decoded tokens:', get_decoded_tokens(text))
    print('token length:', len(get_decoded_tokens(text)))
    # print('token length:', len(enc.encode(text)))

def get_decoded_tokens(text):
    return [token.decode('utf-8', errors='replace') for token in tokenizer.tokenize(text)]

if __name__ == '__main__':
    tokenize_test('안녕하세요')
    tokenize_test('유재석 강호동 박완규')
    tokenize_test('안녕하세요. 고려대학교 구름 (KULLM) 토크나이저 입니다.')