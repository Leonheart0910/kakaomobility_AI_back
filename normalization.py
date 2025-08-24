import re

place_value1_ko = ("", "십", "백", "천")
place_value2_ko = ("", "만", "억", "조", "경")
number_dic_ko = ("", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")


def split_number(number, n):
    """number를 n자리씩 끊어서 리스트로 반환한다."""
    res = []
    div = 10**n
    while number > 0:
        number, remainder = divmod(number, div)
        res.append(remainder)
    return res


def convert_lt_10000(number, delimiter):
    """10000 미만의 수를 한글로 변환한다.
    delimiter가 ''이면 1을 무조건 '일'로 바꾼다."""
    res = ""
    for place, digit in enumerate(split_number(number, 1)):
        if not digit:
            continue
        if delimiter and digit == 1 and place != 0:
            num = ""
        else:
            num = number_dic_ko[digit]
        res = num + place_value1_ko[place] + res
    return res


def number_to_word_ko(number, delimiter=" "):
    """0 이상의 number를 한글로 바꾼다.
    delimiter를 ''로 지정하면 1을 '일'로 바꾸고 공백을 넣지 않는다."""
    if number == 0:
        return "영"
    word_list = []
    for place, digits in enumerate(split_number(number, 4)):
        if word := convert_lt_10000(digits, delimiter):
            word += place_value2_ko[place]
            word_list.append(word)
    res = delimiter.join(word_list[::-1])
    if delimiter and 10000 <= number < 20000:
        res = res[1:]
    return res


def convert_numbers_in_string_to_korean(input_string):

    input_string = re.sub(r'\(.*?\)', '', input_string)

    replacements = {"#": "샵", "@": "골뱅이", "&": "앤드", "=": "는"}
    input_string = "".join(replacements.get(c, c) for c in input_string)
    input_string = re.sub(r"[^\uAC00-\uD7A3a-zA-Z\s0-9]", "", input_string)
    output_string = ""
    current_number = ""

    for char in input_string:
        if char.isdigit():
            current_number += char
        else:
            if current_number:
                korean_number = number_to_word_ko(int(current_number))
                output_string += korean_number
                current_number = ""

            output_string += char

    if current_number:
        korean_number = number_to_word_ko(int(current_number))
        output_string += korean_number

    return output_string