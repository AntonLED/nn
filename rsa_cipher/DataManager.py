import rsa


class DataManager():
    
    def __init__(self, nbits=1024) -> None:
        self.nbits = nbits

    def generate_dataset(self, fname: str, length: int) -> None:
        ROOT = "/home/anton/nn/rsa_cipher/datasets/"
        try:
            input = open(ROOT + fname + "_input.txt", "w")
            output = open(ROOT + fname + "_output.txt", "w")
            for _ in range(length):
                pub, priv = rsa.newkeys(nbits=self.nbits)
                pub_vec = list(pub.save_pkcs1(format="DER"))
                priv_vec = list(priv.save_pkcs1(format="DER"))

                input_line = ""
                output_line = ""
                for byte in pub_vec: input_line += str(byte) + " "
                for byte in priv_vec: output_line += str(byte) + " "

                input.write(input_line[:-1] + "\n")
                output.write(output_line[:-1] + "\n")
                
        except Exception as ex:
            print(f"Raised exception: {ex}")
        finally:
            input.close()
            output.close()
