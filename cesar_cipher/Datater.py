import random, string
import os


class Datater():
    class modes():
        utf = list
        string = str

    def __init__(self, shift) -> None:
        self.shift = shift

    def generate_dataset(self, fname: str, max_length: int, size: int) -> None:
        PREF = "/home/anton/nn/cesar_cipher/datasets/"
        with open(PREF + fname + "_outputs.txt", "w") as y_file:
            for _ in range(size):
                y_file.write(''.join(random.choices(string.ascii_uppercase + string.digits, k = random.randint(1, max_length))) + "\n")
        with open(PREF + fname + "_outputs.txt", "r") as y_file:
            with open(PREF + fname + "_inputs.txt", "w") as x_file:
                for line in y_file.readlines():
                    x_file.write(self.encrypt(line[:-1], mode=Datater.modes.string) + "\n")

    def encrypt(self, message: str or list, mode: type[str] or type[list]) -> str or list[int]:
        if isinstance(message, str):
            if mode == str:
                result = ""
                encr_message = self._encrypt(message)
                for code in encr_message:
                    result += chr(code)
                return result
            elif mode == list:
                return self._encrypt(message)
        elif isinstance(message, list):
            if mode == str:
                result = ""
                for code in message:
                    result += chr((code + self.shift) % 128)
                return result
            elif mode == list:
                return [(code + self.shift) % 128 for code in message]

    def decrypt(self, message: str or list, mode: type[str] or type[list]) -> str or list[int]:
        if isinstance(message, str):
            if mode == str:
                return self._decrypt([ord(symb) for symb in list(message)])
            elif mode == list:
                return [ord(symb) for symb in self._decrypt([ord(symb) for symb in list(message)])]
        elif isinstance(message, list):
            if mode == str:
                return self._decrypt(message)
            elif mode == list:
                return [ord(symb) for symb in self._decrypt(message)]
            
    @staticmethod
    def string_to_utf_vec(string: str) -> list[int]:
        return [ord(symb) for symb in list(string)]

    @staticmethod 
    def utf_vec_to_string(vec: list) -> str:
        result = ""
        for code in vec:
            result += chr(int(code + 0.5))
        return result

    def _encrypt(self, message: str) -> list[int]:
        return [(ord(symb) + self.shift) % 128 for symb in list(message)]

    def _decrypt(self, message: list) -> str:
        result = ""
        for code in message:
            result += chr((int(code + 0.5) - self.shift) % 128)
        return result


